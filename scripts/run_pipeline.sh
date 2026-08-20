#!/usr/bin/env bash
# C-AIMMS (IterRet + Delta-Mem) on LoCoMo -- workstation port of
# run_full_pipeline.slurm, with no SLURM involved.
#
#   bash run_pipeline.sh --smoke     # sample 0 only, 152 questions, ~1h
#   bash run_pipeline.sh             # all 10 samples, 1540 questions, 30h+
#
# Run it under tmux. Nothing here requeues the job if your ssh session drops:
#     tmux new -s caimms
#     bash run_pipeline.sh --smoke
#     <ctrl-b d to detach, tmux attach -t caimms to come back>
#
# GPU layout matches the 2-GPU A100 script: vLLM serves Qwen3-4B on GPU 0, the
# eval loads its own copy plus the Delta-Mem adapter on GPU 1. Both fit inside
# 24GB with room to spare (~8GB of bf16 weights each), so unlike the 1-GPU
# smoke variant on the cluster there is no need to cap vLLM's memory hard.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

RUN_ID="$(date +%Y%m%d_%H%M%S)"          # stands in for SLURM_JOB_ID
mkdir -p "${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${CAIMMS_OUTPUT_DIR}/server_${RUN_ID}.log"
RUN_LOG="${CAIMMS_OUTPUT_DIR}/run_${RUN_ID}.log"

if [ "${SMOKE}" = "1" ]; then
    export WORKMEM_MAX_SAMPLES=1
    # Scratch checkpoint, never the real one: rows a smoke test writes would
    # otherwise be picked up by the full run's resume logic and silently skipped.
    export WORKMEM_OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/smoke_results.jsonl"
    rm -f "${WORKMEM_OUTPUT_FILE}"
    MODE="SMOKE (1 sample / 152 questions)"
else
    export WORKMEM_OUTPUT_FILE="${CAIMMS_OUTPUT_DIR}/workmem_iterret_full.jsonl"
    MODE="FULL (10 samples / 1540 questions)"
fi

caimms_activate

echo "=============================================="
echo "  C-AIMMS ${MODE}"
echo "  run  : ${RUN_ID} on $(hostname) at $(date)"
echo "  out  : ${WORKMEM_OUTPUT_FILE}"
echo "  logs : ${RUN_LOG}"
echo "=============================================="

# ── preflight ─────────────────────────────────────────────────────────────────
# Cheap checks up front. The alternative is discovering a missing file after
# vLLM has spent 90s loading weights, or worse, 20 hours in.
fail=0
for f in "${CAIMMS_MODEL_PATH}/config.json" "${CAIMMS_ADAPTER_DIR}/delta_mem_config.json" \
         "${CAIMMS_ADAPTER_DIR}/delta_mem_adapter.pt" "${CAIMMS_DATA_FILE}"; do
    [ -e "$f" ] || { echo "MISSING: $f"; fail=1; }
done
[ "$fail" = "0" ] || { echo "Run download_assets.sh first."; exit 1; }

NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "${NGPU}" -ge 2 ] || { echo "ERROR: need 2 GPUs, found ${NGPU}."; exit 1; }
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

# No scheduler on this box means nobody is holding the GPUs for you. If someone
# else's process is resident, say so rather than OOMing 90 seconds from now.
OTHER="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l)"
if [ "${OTHER}" -gt 0 ]; then
    echo "  NOTE: ${OTHER} compute process(es) already on these GPUs:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /'
    echo "  Continuing in 10s -- ctrl-c to abort."
    sleep 10
fi

# IterRet swallows any MiniLM load failure and silently downgrades to keyword
# overlap, which changes retrieval without changing the exit code. Fail loudly.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
PY

# ── clean up our own stale servers ────────────────────────────────────────────
# -u "$USER": the cluster script's bare pkill was safe inside a private SLURM
# allocation, but this box is shared -- an unscoped pkill would kill someone
# else's vLLM.
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

# ── 1. vLLM on GPU 0 ──────────────────────────────────────────────────────────
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

# ── 2. wait ───────────────────────────────────────────────────────────────────
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

# ── 3. eval on GPU 1 ──────────────────────────────────────────────────────────
echo "[3/3] Starting eval on GPU 1..."
set +e
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.eval_locomo_iterret_mock 2>&1 | tee -a "${RUN_LOG}"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${WORKMEM_OUTPUT_FILE}" 2>/dev/null || echo 0)"
EXPECT=$([ "${SMOKE}" = "1" ] && echo 152 || echo 1540)
echo "=============================================="
echo "  Done at $(date) | exit ${EVAL_EXIT}"
echo "  Rows written: ${ROWS} (expect ${EXPECT})"
echo "  Results: ${WORKMEM_OUTPUT_FILE}"
echo "=============================================="
exit "${EVAL_EXIT}"
