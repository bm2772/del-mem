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
#
# Needs (one-time, see below): ~/caimms/external/MemoryAgentBench (git clone) and
# its scoring deps (nltk, rouge_score, absl-py, defusedxml) installed with
# `pip install --no-deps --target ~/caimms/external/pydeps ...` -- deliberately
# NOT into the shared env, because locomo_protocol switches to stemmed scoring
# whenever nltk is importable.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

GPU="${GPU:-0}"
ARMS="${ARMS:-deltamem base}"
SOURCE="${MAB_SOURCE:-longmemeval_s*}"
LIMIT="${MAB_INPUT_LIMIT:-65536}"
EXT="${CAIMMS_WORKSPACE}/external"
MAB_ROOT="${MAB_ROOT:-${EXT}/MemoryAgentBench}"
OUT="${CAIMMS_OUTPUT_DIR}/mab_official"
SRC_TAG="${SOURCE//\*/star}"

[ -f "${MAB_ROOT}/utils/eval_other_utils.py" ] || { echo "MISSING ${MAB_ROOT} -- git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench.git ${MAB_ROOT}"; exit 1; }
# scoring deps appended LAST so they never shadow the env's own packages
export PYTHONPATH="${PYTHONPATH}:${EXT}/pydeps"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python3 -c "import nltk, rouge_score" || { echo "scoring deps missing in ${EXT}/pydeps"; exit 1; }

FREE="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
[ "${FREE}" -ge 23000 ] || { echo "GPU ${GPU} has only ${FREE} MiB free (a 64k window needs ~23 GB)"; nvidia-smi --id="${GPU}" --query-compute-apps=pid,used_memory --format=csv; exit 1; }
mkdir -p "${OUT}"

for ARM in ${ARMS}; do
    case "${ARM}" in
        deltamem) ADAPTER_ARGS=(--delta-adapter-dir "${CAIMMS_ADAPTER_DIR}") ;;
        base)     ADAPTER_ARGS=() ;;
        *) echo "unknown arm ${ARM}"; exit 1 ;;
    esac
    STEM="${OUT}/${SRC_TAG}_${ARM}_w${LIMIT}"
    echo "================ ${ARM}  window=${LIMIT}  -> ${STEM}.json  ($(date))"
    CUDA_VISIBLE_DEVICES="${GPU}" python3 -u -m deltamem.eval.official_memory_agent_bench \
        --split Accurate_Retrieval --source "${SOURCE}" \
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
    J="${OUT}/${SRC_TAG}_${ARM}_w${LIMIT}.json"
    [ -s "${J}" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], {k: v for k, v in d['averaged_metrics'].items() if k in ('substring_exact_match','exact_match','f1','input_len')}, 'n=', len(d['data']))" "${J}" "${ARM}"
done
