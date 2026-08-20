#!/bin/bash
#SBATCH --job-name=caimms_smoke
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/kbasu/arnavbhatt/workmem_test/outputs/smoke_%j.log

# Smoke-test variant of run_full_pipeline.slurm -- NOT a replacement for it.
# The real full-run script needs 2 GPUs (vLLM server + eval script on separate
# devices, for genuine throughput) and 24h; this exists only for quick
# validation runs (WORKMEM_MAX_SAMPLES=1-2) where a single conversation's
# worth of questions (~40-50 min measured from pipeline_28955.log /
# pipeline_29293.log: ~13-18 sec/question, plus ~60-90s server startup)
# doesn't justify reserving a second GPU for 24h.
#
# Because only 1 GPU is requested, BOTH the vLLM server and the eval script
# run on the same physical device (both see it as CUDA_VISIBLE_DEVICES=0 --
# unlike the 2-GPU script, there is no device index 1 in this allocation, so
# reusing that script's CUDA_VISIBLE_DEVICES=1 line would crash at CUDA init).
# Co-locating two ~4B-parameter models on one 40GB card is only safe because
# vLLM is explicitly capped below its 0.9-utilization default via
# --gpu-memory-utilization -- the production script never sets this flag,
# which is fine there (vLLM gets its own GPU) but would OOM the eval
# process here if left uncapped.

set -euo pipefail
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

DELTA_MEM_ROOT="/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem"
ITERRET_ROOT="/home/kbasu/arnavbhatt/workmem_test/IterRet"
OUTPUT_DIR="/home/kbasu/arnavbhatt/workmem_test/outputs"
SERVER_LOG="${OUTPUT_DIR}/smoke_server_${SLURM_JOB_ID}.log"
MAX_WAIT=300
INTERVAL=10

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  C-AIMMS SMOKE TEST | Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  (1-GPU co-located variant -- see script header before reusing for a full run)"
echo "=============================================="

source /home/kbasu/arnavbhatt/miniconda3/etc/profile.d/conda.sh
conda activate workmem
export PYTHONPATH="${DELTA_MEM_ROOT}:${ITERRET_ROOT}:${PYTHONPATH:-}"
cd "${DELTA_MEM_ROOT}"

echo "[1/3] Starting vLLM server on the shared GPU (memory-capped to leave room for the eval process)..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507 \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.45 \
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

echo "[3/3] Starting eval on the SAME GPU (device 0 here -- co-located with the server, not a second GPU)..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m deltamem.workmem.eval_locomo_iterret_mock
EVAL_EXIT=$?

echo "=============================================="
echo "  Done at: $(date) | Exit: ${EVAL_EXIT}"
echo "=============================================="

kill ${SERVER_PID} 2>/dev/null || true
exit ${EVAL_EXIT}
