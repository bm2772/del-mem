#!/usr/bin/env bash
# Paired A/B: does OSAM Phase 1 at message_mean (SSW) hurt, given a TSW-trained
# adapter? Same layout as run_pipeline.sh (vLLM on GPU 0, eval on GPU 1), but it
# runs deltamem.workmem.ab_write_granularity instead of the main eval.
#
#   bash run_ab_granularity.sh          # sample 0 only  (~152 questions)
#   bash run_ab_granularity.sh 3        # samples 0-2    (~460 questions)
#
# Run it under tmux -- nothing here restarts the job if your ssh session drops:
#     tmux new -s ab
#     bash run_ab_granularity.sh
#     <ctrl-b d to detach, tmux attach -t ab to come back>
#
# The checkpoint is resumable: re-running skips questions already measured.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

export AB_MAX_SAMPLES="${1:-1}"
export AB_ARMS="${AB_ARMS:-token,message_mean}"

# MUST be absolute and MUST live in CAIMMS_OUTPUT_DIR. The A/B derives its graph
# cache from this path's parent (graph_cache/ next to it), so pointing it
# anywhere else silently rebuilds every conversation graph from ~600 vLLM calls
# instead of loading the ones the full run already produced.
export AB_OUTPUT_FILE="${AB_OUTPUT_FILE:-${CAIMMS_OUTPUT_DIR}/ab_write_granularity.jsonl}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${CAIMMS_OUTPUT_DIR}/ab_server_${RUN_ID}.log"
RUN_LOG="${CAIMMS_OUTPUT_DIR}/ab_run_${RUN_ID}.log"

caimms_activate

echo "=============================================="
echo "  C-AIMMS A/B: Phase 1 write granularity"
echo "  arms : ${AB_ARMS}"
echo "  scope: ${AB_MAX_SAMPLES} conversation(s)"
echo "  run  : ${RUN_ID} on $(hostname) at $(date)"
echo "  out  : ${AB_OUTPUT_FILE}"
echo "  logs : ${RUN_LOG}"
echo "=============================================="

# -- preflight ----------------------------------------------------------------
fail=0
for f in "${CAIMMS_MODEL_PATH}/config.json" "${CAIMMS_ADAPTER_DIR}/delta_mem_config.json" \
         "${CAIMMS_ADAPTER_DIR}/delta_mem_adapter.pt" "${CAIMMS_DATA_FILE}"; do
    [ -e "$f" ] || { echo "MISSING: $f"; fail=1; }
done
[ "$fail" = "0" ] || { echo "Run download_assets.sh first."; exit 1; }

# The whole point of this run is that the two arms take DIFFERENT write paths.
# osam_workmem.py must be the role="user" version -- with role="system" every
# evidence token is stamped message_id=-1, the message_mean mask is empty, and
# both arms fall through to the same token-granularity scan.
if ! grep -q '"role": "user"' "${CAIMMS_ROOT}/delta-Mem/deltamem/workmem/osam_workmem.py"; then
    echo "ERROR: osam_workmem.py still ingests evidence as role='system'."
    echo "       Both arms would take the token path and the A/B would measure nothing."
    echo "       Sync the updated osam_workmem.py before running this."
    exit 1
fi

CACHE_DIR="$(dirname "${AB_OUTPUT_FILE}")/graph_cache"
NCACHED="$(ls "${CACHE_DIR}"/sample_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "graph cache: ${NCACHED} conversation(s) cached in ${CACHE_DIR}"
if [ "${NCACHED}" -lt "${AB_MAX_SAMPLES}" ]; then
    echo "  NOTE: fewer cached graphs than conversations requested -- the missing"
    echo "        ones get rebuilt (~600 vLLM calls each, roughly +10 min apiece)."
fi

NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "${NGPU}" -ge 2 ] || { echo "ERROR: need 2 GPUs, found ${NGPU}."; exit 1; }
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

OTHER="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l)"
if [ "${OTHER}" -gt 0 ]; then
    echo "  NOTE: ${OTHER} compute process(es) already on these GPUs:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /'
    echo "  Continuing in 10s -- ctrl-c to abort."
    sleep 10
fi

# IterRet swallows a MiniLM load failure and downgrades to keyword overlap
# without changing the exit code. Fail loudly instead.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
PY

# -- clean up our own stale servers -------------------------------------------
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

cd "${CAIMMS_ROOT}/delta-Mem"

# -- 1. vLLM on GPU 0 ---------------------------------------------------------
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

# -- 2. wait ------------------------------------------------------------------
MAX_WAIT=600; INTERVAL=10; ELAPSED=0
echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
until curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Tail of ${SERVER_LOG}:"
        tail -40 "${SERVER_LOG}"
        exit 1
    fi
    sleep ${INTERVAL}; ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: server did not come up within ${MAX_WAIT}s."
        tail -40 "${SERVER_LOG}"
        exit 1
    fi
    echo "      ...${ELAPSED}s"
done
echo "      server online."

# -- 3. A/B on GPU 1 ----------------------------------------------------------
echo "[3/3] Starting paired A/B on GPU 1..."
set +e
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.ab_write_granularity 2>&1 | tee -a "${RUN_LOG}"
AB_EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${AB_OUTPUT_FILE}" 2>/dev/null || echo 0)"
echo "=============================================="
echo "  Done at $(date) | exit ${AB_EXIT}"
echo "  Paired rows: ${ROWS}"
echo "  Results: ${AB_OUTPUT_FILE}"
echo "  Log:     ${RUN_LOG}"
echo "=============================================="
exit "${AB_EXIT}"
