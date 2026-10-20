#!/usr/bin/env bash
# delta-mem vs base model on MemoryAgentBench with the delta-mem AUTHORS' OWN
# harness (deltamem/eval/official_memory_agent_bench.py): their prompt layout
# (system message + memorize template + query template), one model.generate per
# question (delta-mem writes every token as it reads, state not reset between
# questions), greedy decoding, max 50 new tokens, and MemoryAgentBench's official
# scoring code (utils/eval_other_utils.py from HUST-AI-HYZ/MemoryAgentBench).
#
# The one forced difference from the paper: the prompt window. The paper fed up
# to the model's context window (~258k tokens) on 80 GB GPUs; a 24 GB 4090 fits
# 64k (measured: 65,536 tokens peaks at 22.9 GB with delta-mem, 81,920 OOMs).
#
#   bash scripts/run_mab_official.sh                       # both models, 64k window, GPU 0
#   ARMS=base GPU=1 MAB_INPUT_LIMIT=32768 bash scripts/run_mab_official.sh
#   MAB_SOURCE=eventqa_65536 ...                 # other MemoryAgentBench sources
#   MAB_SOURCE=factconsolidation_sh_32k ...      # (Conflict_Resolution split picked automatically)
#   DELTAMEM_LEGACY_TOKEN_MASK=1 ARMS=deltamem ...   # upstream token-mask logic (files: _legacymask)
#   MAB_MAX_TEST_SAMPLES=1 ...                   # first history only
# Waits (polling) for the GPU to have ~23 GB free, so it can be queued on a shared box.
#
# Needs (one-time, see below): ~/caimms/external/MemoryAgentBench (git clone) and
# its scoring deps (nltk, rouge_score, absl-py, defusedxml) installed with
# `pip install --no-deps --target ~/caimms/external/pydeps ...` -- deliberately
# NOT into the shared env, because locomo_protocol switches to stemmed scoring
# whenever nltk is importable.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
source "${HERE}/gpu_wait.sh"
caimms_activate

GPU="${GPU:-0}"
ARMS="${ARMS:-deltamem base}"
SOURCE="${MAB_SOURCE:-longmemeval_s*}"
LIMIT="${MAB_INPUT_LIMIT:-65536}"
EXT="${CAIMMS_WORKSPACE}/external"
MAB_ROOT="${MAB_ROOT:-${EXT}/MemoryAgentBench}"
OUT="${CAIMMS_OUTPUT_DIR}/mab_official"
SRC_TAG="${SOURCE//\*/star}"
# MemoryAgentBench split that holds the source
case "${SOURCE}" in
    factconsolidation*) SPLIT="Conflict_Resolution" ;;
    *) SPLIT="${MAB_SPLIT_NAME:-Accurate_Retrieval}" ;;
esac
# Optional: first N histories only (--max-test-samples)
EXTRA_ARGS=()
[ -n "${MAB_MAX_TEST_SAMPLES:-}" ] && EXTRA_ARGS+=(--max-test-samples "${MAB_MAX_TEST_SAMPLES}") && SRC_TAG="${SRC_TAG}_n${MAB_MAX_TEST_SAMPLES}"
# DELTAMEM_LEGACY_TOKEN_MASK=1 -> upstream token-mask logic (see delta_impl.py)
MASK_TAG="$([ "${DELTAMEM_LEGACY_TOKEN_MASK:-0}" = "1" ] && echo "_legacymask")"
export DELTAMEM_LEGACY_TOKEN_MASK="${DELTAMEM_LEGACY_TOKEN_MASK:-0}"

[ -f "${MAB_ROOT}/utils/eval_other_utils.py" ] || { echo "MISSING ${MAB_ROOT} -- git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench.git ${MAB_ROOT}"; exit 1; }
# scoring deps appended LAST so they never shadow the env's own packages
export PYTHONPATH="${PYTHONPATH}:${EXT}/pydeps"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python3 -c "import nltk, rouge_score" || { echo "scoring deps missing in ${EXT}/pydeps"; exit 1; }

# A 64k window needs ~23 GB: wait (polling) until the shared GPU is free.
wait_for_gpu "${GPU}" 23000 "${MAX_WAIT_HOURS:-24}" || exit 1
mkdir -p "${OUT}"

for ARM in ${ARMS}; do
    case "${ARM}" in
        deltamem) ADAPTER_ARGS=(--delta-adapter-dir "${CAIMMS_ADAPTER_DIR}") ;;
        base)     ADAPTER_ARGS=() ;;
        *) echo "unknown arm ${ARM}"; exit 1 ;;
    esac
    STEM="${OUT}/${SRC_TAG}_${ARM}${MASK_TAG}_w${LIMIT}"
    echo "================ ${ARM}  window=${LIMIT}  -> ${STEM}.json  ($(date))"
    CUDA_VISIBLE_DEVICES="${GPU}" python3 -u -m deltamem.eval.official_memory_agent_bench \
        --split "${SPLIT}" --source "${SOURCE}" "${EXTRA_ARGS[@]}" \
        --model-path "${CAIMMS_MODEL_PATH}" "${ADAPTER_ARGS[@]}" \
        --input-length-limit "${LIMIT}" --eval-batch-size 1 \
        --external-memory-agent-bench-root "${MAB_ROOT}" \
        --hub-cache-dir "${HF_HOME}/hub" --datasets-cache-dir "${HF_HOME}/datasets" \
        --output-json "${STEM}.json" --output-jsonl "${STEM}.jsonl" \
        2>&1 | tee -a "${STEM}.log"
    echo "================ ${ARM} done ($(date)) exit ${PIPESTATUS[0]}"
done

echo
for ARM in ${ARMS}; do
    J="${OUT}/${SRC_TAG}_${ARM}${MASK_TAG}_w${LIMIT}.json"
    [ -s "${J}" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], {k: v for k, v in d['averaged_metrics'].items() if k in ('substring_exact_match','exact_match','f1','eventqa_recall','input_len')}, 'n=', len(d['data']))" "${J}" "${ARM}"
done
