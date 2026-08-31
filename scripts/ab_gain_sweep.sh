#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Test-time OSAM gain sweep (NO retraining). The adapter was trained at
# online_gain=0.05, giving a ~2.9% delta_o correction. OSAM_DELTA_GAIN scales
# that trained correction at inference, so we can probe whether a larger gain
# lets the S-only path (OSAM_EVIDENCE_IN_PROMPT=0) recover any of the -27 F1.
#
# Gain -> emulated online_gain:  1->0.05  2->0.10  4->0.20  10->0.50
#
# This is a cheap approximation of retraining: it rescales the SAME trained
# projections rather than learning new ones. If a higher gain clearly helps
# S-only F1 (and delta_o_ratio rises without garbling output), that justifies a
# real retrain (--online-gain / --episode-recent-messages, see the handoff).
#
# Run on a GPU box:  N=1 bash scripts/ab_gain_sweep.sh
# Knobs: N (samples/point, default 1), GAINS (default "1 2 4 10"), VLLM_PORT.
# ---------------------------------------------------------------------------
set -euo pipefail
source env.sh
caimms_activate

N="${N:-1}"
GAINS="${GAINS:-1 2 4 10}"
OUT="${CAIMMS_OUTPUT_DIR:-outputs}"; mkdir -p "$OUT"
EVAL="deltamem.workmem.eval_locomo_iterret_mock"
PY="${PY:-python3}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_LOG="$OUT/vllm_gain.log"
export CAIMMS_VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ── vLLM graph/retrieval server on GPU 0 ─────────────────────────────────────
STARTED_VLLM=0
if curl -sf "${CAIMMS_VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[gain] vLLM already up on :${VLLM_PORT}, reusing it."
else
    echo "[gain] starting vLLM on GPU 0 (port ${VLLM_PORT}); log ${SERVER_LOG} ..."
    CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
        --model "${CAIMMS_MODEL_PATH}" --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
        --port "${VLLM_PORT}" --dtype bfloat16 --max-model-len 8192 \
        --gpu-memory-utilization 0.85 --enforce-eager --disable-log-requests \
        > "${SERVER_LOG}" 2>&1 &
    VLLM_PID=$!; STARTED_VLLM=1
    trap '[ "$STARTED_VLLM" = 1 ] && kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM
    ELAPSED=0
    until curl -sf "${CAIMMS_VLLM_BASE_URL}/models" >/dev/null 2>&1; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "[gain] vLLM died; tail:"; tail -25 "$SERVER_LOG"; exit 1; }
        sleep 10; ELAPSED=$((ELAPSED + 10)); [ $ELAPSED -ge 600 ] && { echo "[gain] vLLM timeout"; tail -25 "$SERVER_LOG"; exit 1; }
    done
    echo "[gain] vLLM online."
fi

echo "[gain] sweep GAINS=[$GAINS]  N=$N/point  (S-only path, OSAM_EVIDENCE_IN_PROMPT=0)"
for g in $GAINS; do
    emu=$(python3 -c "print(f'{0.05*$g:.3f}')")
    out="$OUT/gain_${g}x_evidenceOFF.jsonl"
    echo "[gain] === OSAM_DELTA_GAIN=${g}x (~online_gain ${emu}) -> ${out} ==="
    CUDA_VISIBLE_DEVICES=1 OSAM_EVIDENCE_IN_PROMPT=0 OSAM_DELTA_GAIN="$g" \
      WORKMEM_MAX_SAMPLES="$N" WORKMEM_OUTPUT_FILE="$out" "$PY" -u -m "$EVAL"
    "$PY" scripts/score_calculator.py "$out"
done
echo "[gain] done. Compare the FINAL SCORES per gain above; also inspect each"
echo "       file's osam_contribution.mean_delta_o_ratio (should scale ~linearly with gain)."
