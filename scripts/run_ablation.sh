#!/usr/bin/env bash
# C-AIMMS ABLATION: IterRet retrieval -> plain context -> BASE Qwen3-4B.
# No Delta-Mem adapter, no OSAM. Isolates what IterRet contributes on its own.
#
#   bash scripts/run_ablation.sh 1      # 1 conversation (~150q), quick
#   bash scripts/run_ablation.sh        # all 10 conversations, long
#
# Compare the result against the OSAM pipeline's score on the same questions:
#   higher here  -> OSAM compression is losing information
#   lower here   -> Delta-Mem adds value beyond plain context
#
# Reuses the shared graph cache in $CAIMMS_OUTPUT_DIR/graph_cache, so no graph
# is rebuilt if the main pipeline already produced it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

export ABLATION_MAX_SAMPLES="${1:-}"
[ -z "${ABLATION_MAX_SAMPLES}" ] && unset ABLATION_MAX_SAMPLES

RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${CAIMMS_OUTPUT_DIR}/ablation_server_${RUN_ID}.log"
RUN_LOG="${CAIMMS_OUTPUT_DIR}/ablation_run_${RUN_ID}.log"
export CAIMMS_ABLATION_OUTPUT="${CAIMMS_ABLATION_OUTPUT:-${CAIMMS_OUTPUT_DIR}/workmem_ablation_direct.jsonl}"

caimms_activate

echo "=============================================="
echo "  C-AIMMS ABLATION (IterRet only, NO OSAM)"
echo "  scope : ${ABLATION_MAX_SAMPLES:-all 10} conversation(s)"
echo "  out   : ${CAIMMS_ABLATION_OUTPUT}"
echo "  logs  : ${RUN_LOG}"
echo "=============================================="

# -- preflight --------------------------------------------------------------
fail=0
for f in "${CAIMMS_MODEL_PATH}/config.json" "${CAIMMS_DATA_FILE}"; do
    [ -e "$f" ] || { echo "MISSING: $f"; fail=1; }
done
[ "$fail" = "0" ] || { echo "Run scripts/download_assets.sh first."; exit 1; }

NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "${NGPU}" -ge 2 ] || { echo "ERROR: need 2 GPUs, found ${NGPU}."; exit 1; }

# IterRet silently downgrades to keyword overlap if MiniLM won't load, which
# changes retrieval without changing the exit code. Fail loudly instead.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
PY

NCACHED="$(ls "${CAIMMS_OUTPUT_DIR}/graph_cache"/sample_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "graph cache: ${NCACHED} conversation(s) cached -- these will NOT be rebuilt"

# -- vLLM on GPU 0 ----------------------------------------------------------
pkill -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping vLLM (pid ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[1/3] Starting vLLM on GPU 0 (port ${VLLM_PORT})..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "${CAIMMS_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --disable-log-requests \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "      pid ${SERVER_PID}, log ${SERVER_LOG}"

MAX_WAIT=600; INTERVAL=10; ELAPSED=0
echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
until curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Tail of ${SERVER_LOG}:"; tail -40 "${SERVER_LOG}"; exit 1
    fi
    sleep ${INTERVAL}; ELAPSED=$((ELAPSED + INTERVAL))
    [ ${ELAPSED} -ge ${MAX_WAIT} ] && { echo "ERROR: server did not come up."; tail -40 "${SERVER_LOG}"; exit 1; }
    echo "      ...${ELAPSED}s"
done
echo "      server online."

# -- ablation on GPU 1 ------------------------------------------------------
echo "[3/3] Starting ablation on GPU 1 (base model, no adapter)..."
set +e
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.eval_locomo_ablation 2>&1 | tee -a "${RUN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${CAIMMS_ABLATION_OUTPUT}" 2>/dev/null || echo 0)"
echo "=============================================="
echo "  Done at $(date) | exit ${EXIT} | rows: ${ROWS}"
echo "  Results: ${CAIMMS_ABLATION_OUTPUT}"
echo "  Score:   python3 scripts/score_calculator.py ${CAIMMS_ABLATION_OUTPUT}"
echo "=============================================="
exit "${EXIT}"
