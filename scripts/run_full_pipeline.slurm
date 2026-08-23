#!/bin/bash
#SBATCH --job-name=caimms_iterret
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/home/kbasu/arnavbhatt/workmem_test/outputs/pipeline_%j.log

set -euo pipefail
# Kill zombie vLLM from any previous preempted job
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Kill any zombie vLLM from previous preempted job
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

# Resolve from env.sh rather than hardcoding. ITERRET_ROOT previously pointed at
# a sibling .../workmem_test/IterRet, which is now IterRet.RETIRED (IterRet is
# vendored in the repo) -- this job would have died on `import iterret` after
# allocating 2 GPUs and loading the model.
REPO="${CAIMMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if   [ -f "${REPO}/env.sh" ];    then :
elif [ -f "${REPO}/../env.sh" ]; then REPO="$(cd "${REPO}/.." && pwd)"
else
    echo "ERROR: cannot find env.sh from '${REPO}'."
    echo "       Submit from the repo root:  cd <repo> && sbatch scripts/run_full_pipeline.slurm"
    exit 1
fi
source "${REPO}/env.sh"

DELTA_MEM_ROOT="${CAIMMS_ROOT}/delta-Mem"
OUTPUT_DIR="${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/server_${SLURM_JOB_ID}.log"
MAX_WAIT=600
INTERVAL=10

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  C-AIMMS Pipeline | Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "=============================================="

caimms_activate           # env.sh already set PYTHONPATH to delta-Mem + vendored IterRet
cd "${DELTA_MEM_ROOT}"

echo "[1/3] Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "${CAIMMS_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --enforce-eager \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "      PID: ${SERVER_PID}, log: ${SERVER_LOG}"

echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
ELAPSED=0
until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep ${INTERVAL}
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s."
        tail -30 "${SERVER_LOG}"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
    echo "      ...${ELAPSED}s elapsed"
done
echo "      Server online."

echo "[3/3] Starting eval on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.eval_locomo_iterret_mock
EVAL_EXIT=$?

echo "=============================================="
echo "  Done at: $(date) | Exit: ${EVAL_EXIT}"
echo "=============================================="

kill ${SERVER_PID} 2>/dev/null || true
exit ${EVAL_EXIT}
