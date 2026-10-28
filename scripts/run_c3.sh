#!/usr/bin/env bash
# Ablation C3: IterRet with an experience bank bootstrapped on held-out LoCoMo
# conversations (5-10) vs the empty bank of every published run.
#   1. build the bank (needs graphs for conversations 5-10; cached ones are reused)
#   2. LoCoMo-4 with the bank (compare with A1's episodic/document file)
#   3. LongMemEval with the bank (compare with the B2 evidence-only file)
#
#   bash scripts/run_c3.sh
#   C3_SKIP_LME=1 bash scripts/run_c3.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
O="${CAIMMS_OUTPUT_DIR}"
BANK="${O}/experience_bank_heldout.json"

[ -s "${BANK}" ] || EVAL_MODULE=deltamem.workmem.build_heldout_bank WORKMEM_OUTPUT_FILE="${BANK}" \
    bash "${HERE}/run_pipeline.sh"
[ -s "${BANK}" ] || { echo "[c3] bank was not built"; exit 1; }

ITERRET_BANK_PATH="${BANK}" WORKMEM_MAX_SAMPLES=4 WORKMEM_EVIDENCE_LAYERS=episodic WORKMEM_EVIDENCE_ORDER=document \
    WORKMEM_OUTPUT_FILE="${O}/C3_locomo4_combined_episodic_document_bankheldout.jsonl" bash "${HERE}/run_pipeline.sh"
[ -n "${C3_SKIP_LME:-}" ] || ITERRET_BANK_PATH="${BANK}" TAG="${TAG:-released_B2}" ARMS=combined bash "${HERE}/run_mab.sh"
