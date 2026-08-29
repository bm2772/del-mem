#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OSAM A/B: evidence-in-prompt (default) vs evidence-only-through-S (paper §1.5).
# Runs the workmem eval twice, same everything except OSAM_EVIDENCE_IN_PROMPT,
# then scores both. Arm B (=0) is the spec-faithful path and also the one that
# fixes the osam_contribution 0.0 reading (fresh Phase-2 ingest -> valid mask).
#
# Run on a GPU box (resiliente-2003 / Mahamathi), from the repo root:
#   bash scripts/ab_evidence.sh
#
# Knobs (env):
#   N               samples per arm (default 4; keep small -- this is a probe)
#   CAIMMS_MODEL_PATH / CAIMMS_ADAPTER_DIR   backbone + TRAINED adapter
#
# NOTE the adapter must be trained FOR the backbone in CAIMMS_MODEL_PATH.
# The shipped adapter is Qwen3-4B-only; a 1.7B run needs a 1.7B-trained adapter
# (set both vars to point at it), otherwise load_state_dict will shape-mismatch.
# ---------------------------------------------------------------------------
set -euo pipefail
source env.sh

N="${N:-4}"
OUT="${CAIMMS_OUTPUT_DIR:-outputs}"
mkdir -p "$OUT"
EVAL="deltamem.workmem.eval_locomo_iterret_mock"
# The caimms env exposes python3 (not always `python`) -- match run_pipeline.sh.
PY="${PY:-python3}"

echo "[ab] backbone=${CAIMMS_MODEL_PATH##*/}  adapter=${CAIMMS_ADAPTER_DIR##*/}  N=$N/arm"

echo "[ab] === Arm A: WITH evidence in prompt (OSAM_EVIDENCE_IN_PROMPT=1) ==="
OSAM_EVIDENCE_IN_PROMPT=1 WORKMEM_MAX_SAMPLES="$N" \
  WORKMEM_OUTPUT_FILE="$OUT/ab_with_evidence.jsonl" \
  "$PY" -m "$EVAL"

echo "[ab] === Arm B: evidence ONLY through S (OSAM_EVIDENCE_IN_PROMPT=0) ==="
OSAM_EVIDENCE_IN_PROMPT=0 WORKMEM_MAX_SAMPLES="$N" \
  WORKMEM_OUTPUT_FILE="$OUT/ab_without_evidence.jsonl" \
  "$PY" -m "$EVAL"

echo "[ab] === scores ==="
echo "-- WITH evidence --"    && "$PY" scripts/score_calculator.py "$OUT/ab_with_evidence.jsonl"
echo "-- WITHOUT evidence --" && "$PY" scripts/score_calculator.py "$OUT/ab_without_evidence.jsonl"
echo "[ab] Also check osam_contribution: Arm B rows should now carry a REAL"
echo "     delta_o ratio (~0.02), not 0.0 -- that confirms the B3 mask fix."
