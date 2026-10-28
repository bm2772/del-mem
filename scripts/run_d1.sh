#!/usr/bin/env bash
# Test-plan D1: IterRet with a second model family for graph building, search
# and answering (LoCoMo-4, final evidence format). Answers come straight from the
# served model (WORKMEM_ANSWER_BACKEND=vllm, no delta-mem: the adapter is
# Qwen-only), so the Qwen arm without delta-mem is run too, as the like-for-like
# reference. Graphs are rebuilt per family (WORKMEM_GRAPH_CACHE_DIR).
#
#   D1_MODEL=meta-llama/Llama-3.1-8B-Instruct bash scripts/run_d1.sh   # gated: hf auth login first
#   D1_MODEL=mistralai/Mistral-7B-Instruct-v0.3 bash scripts/run_d1.sh
#   D1_SKIP_QWEN=1 ...                                                 # family arm only
#
# Weights go to $CAIMMS_WORKSPACE/models/<repo name>; downloaded if missing.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate
O="${CAIMMS_OUTPUT_DIR}"
: "${D1_MODEL:?set D1_MODEL to a Hugging Face repo id}"
NAME="${D1_MODEL##*/}"
DIR="${CAIMMS_WORKSPACE}/models/${NAME}"
COMMON=(WORKMEM_ANSWER_BACKEND=vllm WORKMEM_MAX_SAMPLES="${D1_SAMPLES:-4}"
        WORKMEM_EVIDENCE_LAYERS=episodic WORKMEM_EVIDENCE_ORDER=document)

if [ -z "${D1_SKIP_QWEN:-}" ]; then
    env "${COMMON[@]}" WORKMEM_OUTPUT_FILE="${O}/D1_locomo${D1_SAMPLES:-4}_Qwen3-4B_nodeltamem.jsonl" \
        bash "${HERE}/run_pipeline.sh"
fi

[ -e "${DIR}/config.json" ] || hf download "${D1_MODEL}" --local-dir "${DIR}" \
    --exclude "original/*" "*.pth" || { echo "[d1] download of ${D1_MODEL} failed"; exit 1; }
env "${COMMON[@]}" CAIMMS_MODEL_PATH="${DIR}" CAIMMS_SERVED_MODEL_NAME="${D1_MODEL}" \
    WORKMEM_GRAPH_CACHE_DIR="${O}/graph_cache_${NAME}" \
    WORKMEM_OUTPUT_FILE="${O}/D1_locomo${D1_SAMPLES:-4}_${NAME}.jsonl" bash "${HERE}/run_pipeline.sh"
