#!/usr/bin/env bash
# IterRet + delta-mem on MemoryAgentBench (default: LongMemEval, "longmemeval_s*"),
# histories far longer than the GPU can attend over. One run per mode through
# run_pipeline.sh (vLLM + eval). See deltamem/workmem/eval_mab_iterret.py.
#
#   MAB_MAX_CONTEXTS=1 MAB_MAX_QUESTIONS=10 bash scripts/run_mab.sh   # smoke test first
#   bash scripts/run_mab.sh                                            # all 4 modes, all histories
#   ARMS="truncated truncated_fullS" bash scripts/run_mab.sh
#   MAB_PROMPT_TOKENS=16384 ...       # smaller recent window if 32k runs out of memory
#   CAIMMS_ADAPTER_DIR=... TAG=iterret_w1024 ...                       # another adapter
#   MAB_SPLIT=fixed ...               # generic chunks instead of dated LongMemEval sessions
#   OSAM_PHASE1_GRANULARITY=token ... # write every token into S (how the adapter was trained)
#   OSAM_DELTA_GAIN=0 ARMS=truncated ...   # base model: delta-mem's corrections zeroed
#   ARMS=combined_recent MAB_RECENT_TOKENS=8192 ...   # IterRet evidence + recent window (A2)
#   MAB_RECENT_ALIGN=question_date ...     # window ends at the question's Current Date, not the history's end
#   ITERRET_MAX_ITERATIONS=1 ...           # single-round retrieval (ablation C1)
#   RETRIEVER=flat FLAT_TOP_K=13 ...       # embedding top-k, no graph (ablation C2)
#   ITERRET_RESOLVE_DATES=0 ...            # no date resolution (C4)
#   DISABLE_TAG_EMBEDDER_FUSION=1 ...      # keyword-only facet ranking (C5)
#   ITERRET_BANK_PATH=$CAIMMS_OUTPUT_DIR/experience_bank_heldout.json ...  # bootstrapped bank (C3)
#
# Modes: truncated (recent window only) | truncated_fullS (recent window + whole
# history in S) | combined (IterRet evidence in S and prompt) | hybrid (whole
# history in S, IterRet evidence in prompt).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"

export MAB_SOURCE="${MAB_SOURCE:-longmemeval_s*}"
ARMS="${ARMS:-truncated truncated_fullS combined hybrid}"
TAG="${TAG:-released}"
SRC_TAG="${MAB_SOURCE//\*/star}"
export MAB_SPLIT="${MAB_SPLIT:-sessions}"
SUFFIX="_${MAB_SPLIT}"
[ -n "${MAB_MAX_CONTEXTS:-}" ] && [ "${MAB_MAX_CONTEXTS}" != "0" ] && SUFFIX="${SUFFIX}_c${MAB_MAX_CONTEXTS}"
[ -n "${MAB_MAX_QUESTIONS:-}" ] && [ "${MAB_MAX_QUESTIONS}" != "0" ] && SUFFIX="${SUFFIX}_q${MAB_MAX_QUESTIONS}"
[ "${MAB_PROMPT_TOKENS:-32768}" = "32768" ] || SUFFIX="${SUFFIX}_w${MAB_PROMPT_TOKENS}"
# Write granularity of evidence/history into S: message_mean (default) | token.
[ "${OSAM_PHASE1_GRANULARITY:-message_mean}" = "message_mean" ] || SUFFIX="${SUFFIX}_${OSAM_PHASE1_GRANULARITY}write"
[ "${OSAM_DELTA_GAIN:-1}" = "1" ] || SUFFIX="${SUFFIX}_gain${OSAM_DELTA_GAIN}"
# Test-plan ablations (defaults keep the original file names)
[ "${MAB_RECENT_TOKENS:-8192}" = "8192" ] || SUFFIX="${SUFFIX}_recent${MAB_RECENT_TOKENS}"
[ "${MAB_RECENT_ALIGN:-end}" = "end" ] || SUFFIX="${SUFFIX}_qdate"
[ "${ITERRET_MAX_ITERATIONS:-5}" = "5" ] || SUFFIX="${SUFFIX}_rounds${ITERRET_MAX_ITERATIONS}"
[ "${RETRIEVER:-iterret}" = "iterret" ] || SUFFIX="${SUFFIX}_${RETRIEVER}k${FLAT_TOP_K:-12}"
[ "${ITERRET_RESOLVE_DATES:-1}" = "1" ] || SUFFIX="${SUFFIX}_nodates"
[ -z "${DISABLE_TAG_EMBEDDER_FUSION:-}" ] || SUFFIX="${SUFFIX}_nofusion"
[ -z "${ITERRET_BANK_PATH:-}" ] || SUFFIX="${SUFFIX}_bank$(basename "${ITERRET_BANK_PATH}" .json | sed "s/^experience_bank//")"

for ARM in ${ARMS}; do
    case "${ARM}" in combined|truncated|truncated_fullS|hybrid|combined_recent) ;; *) echo "unknown mode '${ARM}'"; exit 1 ;; esac
    OUT="${CAIMMS_OUTPUT_DIR}/mab_${SRC_TAG}_${ARM}_${TAG}${SUFFIX}.jsonl"
    echo "================ MemoryAgentBench ${MAB_SOURCE} mode=${ARM} adapter=${CAIMMS_ADAPTER_DIR} -> ${OUT}"
    EVAL_MODULE=deltamem.workmem.eval_mab_iterret MAB_MODE="${ARM}" WORKMEM_OUTPUT_FILE="${OUT}" \
        bash "${HERE}/run_pipeline.sh" || echo "!! mode ${ARM} exited non-zero -- continuing"
done

echo
echo "================ summary (${MAB_SOURCE}, ${TAG}${SUFFIX})"
for ARM in ${ARMS}; do
    OUT="${CAIMMS_OUTPUT_DIR}/mab_${SRC_TAG}_${ARM}_${TAG}${SUFFIX}.jsonl"
    [ -s "${OUT}" ] || { echo "${ARM}: no rows"; continue; }
    python3 - "${OUT}" "${ARM}" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("prediction") == "" and not r.get("skipped"):
        continue
    rows[(r["ctx_idx"], r["q_idx"])] = r
rows = list(rows.values())
n = max(1, len(rows))
j = [r["judge_correct"] for r in rows if "judge_correct" in r]
print(f"{sys.argv[2]:>16}: n={len(rows)}  substringEM={sum(r['substring_exact_match'] for r in rows) / n:.4f}  "
      f"F1={sum(r['f1'] for r in rows) / n:.4f}" + (f"  judge={sum(j) / len(j):.4f}" if j else ""))
PY
done
