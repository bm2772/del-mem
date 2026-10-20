#!/usr/bin/env bash
# Re-score LongMemEval results with MemoryAgentBench's official LLM-judge prompts
# and an open-source judge (default Qwen2.5-14B-Instruct-AWQ, ~10 GB, one 24 GB
# GPU) instead of GPT-4o. See deltamem/workmem/judge_mab_longmemeval.py.
#
#   bash scripts/judge_mab_longmemeval.sh                     # judges cached_results/mab_official/slim_*
#   DOWNLOAD_JUDGE=1 bash scripts/judge_mab_longmemeval.sh    # first run: fetch the judge model
#   JUDGE_MODEL_ID=Qwen/Qwen2.5-32B-Instruct-AWQ ...          # stronger judge (~20 GB disk)
#   RESULTS="a.jsonl b.json" JUDGE_GPU=1 ...
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
source "${HERE}/gpu_wait.sh"
caimms_activate

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
JUDGE_DIR="${CAIMMS_WORKSPACE}/models/${JUDGE_MODEL_ID##*/}"
JUDGE_GPU="${JUDGE_GPU:-0}"
JUDGE_PORT="${JUDGE_PORT:-8010}"
TAG="${TAG:-${JUDGE_MODEL_ID##*/}}"
RESULTS="${RESULTS:-$(ls "${CAIMMS_ROOT}"/cached_results/mab_official/slim_longmemeval_sstar_*_w65536.jsonl)}"

if [ ! -f "${JUDGE_DIR}/config.json" ]; then
    if [ "${DOWNLOAD_JUDGE:-0}" != "1" ]; then
        echo "Judge model not found at ${JUDGE_DIR}. Re-run with DOWNLOAD_JUDGE=1 to fetch ${JUDGE_MODEL_ID}."
        df -h "${CAIMMS_WORKSPACE}" | tail -1
        exit 1
    fi
    hf download "${JUDGE_MODEL_ID}" --local-dir "${JUDGE_DIR}" \
        || huggingface-cli download "${JUDGE_MODEL_ID}" --local-dir "${JUDGE_DIR}" || exit 1
fi

wait_for_gpu "${JUDGE_GPU}" 18000 "${MAX_WAIT_HOURS:-24}" || exit 1
LOG="${CAIMMS_OUTPUT_DIR}/judge_server_$(date +%Y%m%d_%H%M%S).log"
# own process group, so cleanup also kills vLLM's EngineCore worker
CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_DIR}" --served-model-name judge --port "${JUDGE_PORT}" \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager --disable-log-requests \
    > "${LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- -"${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null; }
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
    curl -sf "http://localhost:${JUDGE_PORT}/v1/models" >/dev/null && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "judge vLLM died:"; tail -20 "${LOG}"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${JUDGE_PORT}/v1/models" >/dev/null || { echo "judge vLLM did not come up"; tail -20 "${LOG}"; exit 1; }
echo "judge ${JUDGE_MODEL_ID} online on GPU ${JUDGE_GPU}"

export PYTHONPATH="${PYTHONPATH}:${CAIMMS_WORKSPACE}/external/pydeps"
# shellcheck disable=SC2086
python3 -u -m deltamem.workmem.judge_mab_longmemeval --results ${RESULTS} \
    --judge-url "http://localhost:${JUDGE_PORT}/v1" --judge-model judge --tag "${TAG}" \
    --mab-root "${CAIMMS_WORKSPACE}/external/MemoryAgentBench"
