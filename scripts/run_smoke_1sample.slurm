#!/bin/bash
#SBATCH --job-name=caimms_smoke1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/home/kbasu/arnavbhatt/workmem_test/outputs/smoke1_%j.log

# Self-contained 1-conversation smoke test. Just `sbatch run_smoke_1sample.slurm`
# -- no --export flags needed, everything is set below.
#
# NOT a substitute for run_full_pipeline.slurm: that one needs 2 GPUs (vLLM
# server and eval on separate devices) and 24h for the full 1540 questions.
# This runs sample 0 only (152 questions), measured at ~43 min end-to-end
# (~13-18 sec/question plus ~60-90s server startup), so 1 GPU and 1.5h is
# ample -- reserving a second GPU for 24h to check 152 questions is waste.
#
# Because only 1 GPU is requested, BOTH the vLLM server and the eval process
# run on the same physical device (both see it as index 0 -- there is no
# device 1 in this allocation, so copying the 2-GPU script's
# CUDA_VISIBLE_DEVICES=1 line here would crash at CUDA init). Co-locating two
# ~4B models on one 40GB card is only safe because vLLM is explicitly capped
# below its 0.9-utilization default via --gpu-memory-utilization; the
# production script never sets that flag (it has a dedicated GPU) and leaving
# it unset here would OOM the eval process.

set -euo pipefail
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

DELTA_MEM_ROOT="/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem"
ITERRET_ROOT="/home/kbasu/arnavbhatt/workmem_test/IterRet"
OUTPUT_DIR="/home/kbasu/arnavbhatt/workmem_test/outputs"
SERVER_LOG="${OUTPUT_DIR}/smoke1_server_${SLURM_JOB_ID}.log"
MAX_WAIT=300
INTERVAL=10

# Sample 0 only.
export WORKMEM_MAX_SAMPLES=1
# Written to SHARED storage, not /tmp: /tmp is node-local, so a previous smoke
# test's output was unreachable from the login node and lost when the job ended.
# Also a scratch filename, so checkpoint-resume in the REAL run can never pick
# these rows up and silently skip those questions.
export WORKMEM_OUTPUT_FILE="${OUTPUT_DIR}/smoke1_results.jsonl"

# Uncomment to measure the IDF ranking change in isolation from the existing
# RRF/MiniLM fusion (see ctc_graph.rank_contents_by_relevance):
# export DISABLE_CONTENT_EMBEDDER_FUSION=1

mkdir -p "${OUTPUT_DIR}"

# Start clean: this file is a scratch checkpoint, and leaving stale rows in it
# would make the eval skip those questions and silently test nothing.
rm -f "${WORKMEM_OUTPUT_FILE}"

echo "=============================================="
echo "  C-AIMMS SMOKE (1 sample) | Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  MAX_SAMPLES=${WORKMEM_MAX_SAMPLES}  OUT=${WORKMEM_OUTPUT_FILE}"
echo "=============================================="

source /home/kbasu/arnavbhatt/miniconda3/etc/profile.d/conda.sh
conda activate workmem
export PYTHONPATH="${DELTA_MEM_ROOT}:${ITERRET_ROOT}:${PYTHONPATH:-}"
cd "${DELTA_MEM_ROOT}"

echo "[1/3] Starting vLLM server on the shared GPU (memory-capped)..."
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

echo "[3/3] Starting eval on the SAME GPU (co-located with the server)..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m deltamem.workmem.eval_locomo_iterret_mock
EVAL_EXIT=$?

echo "=============================================="
echo "  Done at: $(date) | Exit: ${EVAL_EXIT}"
echo "  Rows written: $(wc -l < "${WORKMEM_OUTPUT_FILE}" 2>/dev/null || echo 0) (expect 152)"
echo "=============================================="

kill ${SERVER_PID} 2>/dev/null || true
exit ${EVAL_EXIT}
