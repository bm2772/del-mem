#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OSAM A/B: evidence-in-prompt (default) vs evidence-only-through-S (paper §1.5).
# Runs the workmem eval twice, same everything except OSAM_EVIDENCE_IN_PROMPT,
# then scores both. Arm B (=0) is the spec-faithful path and also the one that
# fixes the osam_contribution 0.0 reading (fresh Phase-2 ingest -> valid mask).
#
# Like run_pipeline.sh, this starts the vLLM graph/retrieval server on GPU 0 and
# runs the answering model (+adapter) on GPU 1. It reuses an already-running
# vLLM if one is up on VLLM_PORT.
#
# Run on a GPU box (resiliente-2003 / Mahamathi), from the repo root:
#   N=4 bash scripts/ab_evidence.sh
#
# Knobs (env): N (samples/arm, default 4), VLLM_PORT (default 8000),
#   CAIMMS_MODEL_PATH / CAIMMS_ADAPTER_DIR (backbone + TRAINED adapter).
# NOTE the adapter must be trained FOR the backbone; the shipped one is 4B-only.
# ---------------------------------------------------------------------------
set -euo pipefail
source env.sh
# env.sh only DEFINES caimms_activate; call it (as run_pipeline.sh does) so the
# workmem conda env is active and python3 sees torch/deltamem.
caimms_activate

N="${N:-4}"
OUT="${CAIMMS_OUTPUT_DIR:-outputs}"
mkdir -p "$OUT"
EVAL="deltamem.workmem.eval_locomo_iterret_mock"
PY="${PY:-python3}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_LOG="$OUT/vllm_ab.log"
# Make the eval's client point at the same port we serve on.
export CAIMMS_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ── vLLM graph/retrieval server on GPU 0 (IterRet's extraction + reflect) ─────
STARTED_VLLM=0
if curl -sf "${CAIMMS_VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[ab] vLLM already up on :${VLLM_PORT}, reusing it."
else
    echo "[ab] starting vLLM on GPU 0 (port ${VLLM_PORT}); log ${SERVER_LOG} ..."
    CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
        --model "${CAIMMS_MODEL_PATH}" --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
        --port "${VLLM_PORT}" --dtype bfloat16 --max-model-len 8192 \
        --gpu-memory-utilization 0.85 --enforce-eager --disable-log-requests \
        > "${SERVER_LOG}" 2>&1 &
    VLLM_PID=$!
    STARTED_VLLM=1
    trap '[ "$STARTED_VLLM" = 1 ] && kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM
    ELAPSED=0
    until curl -sf "${CAIMMS_VLLM_BASE_URL}/models" >/dev/null 2>&1; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "[ab] vLLM died; tail of log:"; tail -25 "$SERVER_LOG"; exit 1; }
        sleep 10; ELAPSED=$((ELAPSED + 10))
        [ $ELAPSED -ge 600 ] && { echo "[ab] vLLM timeout after ${ELAPSED}s"; tail -25 "$SERVER_LOG"; exit 1; }
        echo "      ...waiting ${ELAPSED}s"
    done
    echo "[ab] vLLM online."
fi

echo "[ab] backbone=${CAIMMS_MODEL_PATH##*/}  adapter=${CAIMMS_ADAPTER_DIR##*/}  N=$N/arm"

# ── arms: answering model (+adapter) on GPU 1 ────────────────────────────────
echo "[ab] === Arm A: WITH evidence in prompt (OSAM_EVIDENCE_IN_PROMPT=1) ==="
CUDA_VISIBLE_DEVICES=1 OSAM_EVIDENCE_IN_PROMPT=1 WORKMEM_MAX_SAMPLES="$N" \
  WORKMEM_OUTPUT_FILE="$OUT/ab_with_evidence.jsonl" "$PY" -u -m "$EVAL"

echo "[ab] === Arm B: evidence ONLY through S (OSAM_EVIDENCE_IN_PROMPT=0) ==="
CUDA_VISIBLE_DEVICES=1 OSAM_EVIDENCE_IN_PROMPT=0 WORKMEM_MAX_SAMPLES="$N" \
  WORKMEM_OUTPUT_FILE="$OUT/ab_without_evidence.jsonl" "$PY" -u -m "$EVAL"

echo "[ab] === scores ==="
echo "-- WITH evidence --"    && "$PY" scripts/score_calculator.py "$OUT/ab_with_evidence.jsonl"
echo "-- WITHOUT evidence --" && "$PY" scripts/score_calculator.py "$OUT/ab_without_evidence.jsonl"
echo "[ab] Also check osam_contribution: Arm B rows should now carry a REAL"
echo "     delta_o ratio (~0.02), not 0.0 -- that confirms the B3 mask fix."
