#!/bin/bash
# C-AIMMS ABLATION on SLURM: IterRet retrieval -> plain context -> BASE Qwen3-4B.
# No Delta-Mem adapter, no OSAM. Isolates what IterRet contributes on its own.
#
#   sbatch scripts/run_ablation.slurm            # all 10 conversations
#   sbatch --export=ALL,ABLATION_MAX_SAMPLES=1 scripts/run_ablation.slurm   # 1 conv
#
# Partition is overridable without editing this file:
#   sbatch --partition=ada scripts/run_ablation.slurm
#
#SBATCH --job-name=caimms_ablation
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.log

set -euo pipefail

# Resolve the repo from this script's own location, then let env.sh derive
# every path. The previous version hardcoded ITERRET_ROOT to a sibling
# .../workmem_test/IterRet -- that directory is now IterRet.RETIRED (IterRet is
# vendored inside the repo), so this script would have died on `import iterret`.
# SLURM_SUBMIT_DIR is wherever you ran sbatch from -- accept either the repo
# root or scripts/, and allow an explicit override for anything else.
REPO="${CAIMMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if   [ -f "${REPO}/env.sh" ];    then :
elif [ -f "${REPO}/../env.sh" ]; then REPO="$(cd "${REPO}/.." && pwd)"
else
    echo "ERROR: cannot find env.sh from '${REPO}'."
    echo "       Submit from the repo root:  cd <repo> && sbatch scripts/run_ablation.slurm"
    echo "       or pass it explicitly:      sbatch --export=ALL,CAIMMS_REPO=/path/to/workmem-vertical ..."
    exit 1
fi
source "${REPO}/env.sh"

OUTPUT_DIR="${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/ablation_server_${SLURM_JOB_ID}.log"
RUN_LOG="${OUTPUT_DIR}/ablation_run_${SLURM_JOB_ID}.log"
MAX_WAIT=600      # was 300; vLLM startup can exceed that on a contended node
INTERVAL=10

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  C-AIMMS ABLATION (IterRet only, NO OSAM)"
echo "  Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  repo : ${CAIMMS_ROOT}"
echo "  scope: ${ABLATION_MAX_SAMPLES:-all 10} conversation(s)"
echo "  out  : ${CAIMMS_OUTPUT_DIR}/workmem_ablation_direct.jsonl"
echo "=============================================="

caimms_activate
cd "${CAIMMS_ROOT}/delta-Mem"

NCACHED="$(ls "${OUTPUT_DIR}/graph_cache"/sample_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "graph cache: ${NCACHED} conversation(s) cached -- these will NOT be rebuilt"

# IterRet silently downgrades to keyword overlap if MiniLM won't load, which
# changes retrieval without changing the exit code. Fail loudly instead.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
PY

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
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
    --enforce-eager \
    --disable-log-requests \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Real cause:"
        grep -nE "OutOfMemory|CUDA|Error|Traceback|Address already in use" "${SERVER_LOG}" | tail -15
        echo "--- last 25 lines ---"; tail -25 "${SERVER_LOG}"
        exit 1
    fi
    sleep ${INTERVAL}; ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: server did not come up within ${MAX_WAIT}s."
        grep -nE "OutOfMemory|CUDA|Error|Traceback" "${SERVER_LOG}" | tail -15
        tail -25 "${SERVER_LOG}"
        exit 1
    fi
    echo "      ...${ELAPSED}s"
done
echo "      Server online."

echo "[3/3] Starting ablation on GPU 1 (base model, no adapter)..."
set +e
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.eval_locomo_ablation 2>&1 | tee -a "${RUN_LOG}"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${CAIMMS_OUTPUT_DIR}/workmem_ablation_direct.jsonl" 2>/dev/null || echo 0)"
echo "=============================================="
echo "  Done at $(date) | exit ${EVAL_EXIT} | rows: ${ROWS}"
echo "  Score: python3 ${CAIMMS_ROOT}/scripts/score_calculator.py ${CAIMMS_OUTPUT_DIR}/workmem_ablation_direct.jsonl"
echo "=============================================="
exit "${EVAL_EXIT}"
