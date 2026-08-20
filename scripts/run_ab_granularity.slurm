#!/bin/bash
#SBATCH --job-name=caimms_ab_gran
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/home/kbasu/arnavbhatt/workmem_test/outputs/ab_gran_%j.log

# Paired A/B on the Phase 1 OSAM write granularity: "token" (TSW -- what the
# released adapter was trained with) vs "message_mean" (SSW -- what the
# two-phase design calls for). Just `sbatch run_ab_granularity.slurm`.
#
# Sizing follows run_smoke_1sample.slurm: 1 conversation, 1 GPU, vLLM capped so
# it can share the card with the eval model. The A/B generates TWICE per
# question but retrieves ONCE (retrieval is the expensive part, ~20 vLLM calls),
# so it costs roughly the smoke test plus one extra generation pass -- ~60-75
# min measured against the smoke test's ~43. 2.5h leaves headroom for a cold
# graph build on the first run.
#
# Resumable: --requeue plus the script's own checkpoint means a preempted job
# picks up where it stopped. Set AB_FRESH=1 to discard prior rows instead
# (needed whenever the code under test changed, or the stale rows would be
# reported as if they came from the new code).

set -euo pipefail
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

DELTA_MEM_ROOT="/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem"
ITERRET_ROOT="/home/kbasu/arnavbhatt/workmem_test/IterRet"
OUTPUT_DIR="/home/kbasu/arnavbhatt/workmem_test/outputs"
SERVER_LOG="${OUTPUT_DIR}/ab_gran_server_${SLURM_JOB_ID}.log"
MAX_WAIT=300
INTERVAL=10

export AB_MAX_SAMPLES=1
export AB_ARMS="token,message_mean"     # first entry is the baseline
# Must live in OUTPUT_DIR: the A/B derives its graph cache from this path's
# parent, so keeping it here reuses outputs/graph_cache/sample_0.json and skips
# the ~600-call CTC graph build the main run already paid for.
export AB_OUTPUT_FILE="${OUTPUT_DIR}/ab_write_granularity.jsonl"

mkdir -p "${OUTPUT_DIR}"

if [ "${AB_FRESH:-0}" = "1" ]; then
    echo "AB_FRESH=1 -- discarding any previous rows."
    rm -f "${AB_OUTPUT_FILE}"
fi

echo "=============================================="
echo "  C-AIMMS A/B write granularity | Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "  $(date)"
echo "  ARMS=${AB_ARMS}  SAMPLES=${AB_MAX_SAMPLES}"
echo "  OUT=${AB_OUTPUT_FILE}"
echo "  resuming from: $([ -f "${AB_OUTPUT_FILE}" ] && wc -l < "${AB_OUTPUT_FILE}" || echo 0) rows"
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
# Kill the server on ANY exit path. The other slurm scripts kill it only on the
# success path, but `set -e` means a crashing eval skips that line entirely and
# leaves a GPU-resident vLLM behind for the next job to pkill.
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
echo "      PID: ${SERVER_PID}, log: ${SERVER_LOG}"

echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
ELAPSED=0
until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep ${INTERVAL}
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Server did not start within ${MAX_WAIT}s."
        tail -30 "${SERVER_LOG}"
        exit 1
    fi
    echo "      ...${ELAPSED}s elapsed"
done
echo "      Server online."

echo "[3/3] Running the paired A/B on the SAME GPU..."
AB_EXIT=0
CUDA_VISIBLE_DEVICES=0 python3 -u -m deltamem.workmem.ab_write_granularity || AB_EXIT=$?

echo "=============================================="
echo "  Done at: $(date) | Exit: ${AB_EXIT}"
echo "  Rows: $(wc -l < "${AB_OUTPUT_FILE}" 2>/dev/null || echo 0)"
echo "=============================================="
exit ${AB_EXIT}
