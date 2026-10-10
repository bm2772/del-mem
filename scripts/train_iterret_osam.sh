#!/usr/bin/env bash
# Fine-tune the delta-mem adapter for the WORKMEM `combined` regime: the SAME
# IterRet evidence is written into S and visible in the prompt.
#
# Objective = context_ablation_ce, mixed per step:
#   60%  full_context_plus_state : S primed from the evidence, model attends evidence + question
#                                  (exactly `combined` at inference)
#   20%  state_only              : S primed from the evidence, model sees only the question
#                                  (forces S to carry usable content)
#   20%  full_context_no_state   : evidence visible, S empty (keeps the no-memory path sane)
# The released recipe (context_dropout_ce) never puts the written text in the
# visible context, which is the mismatch this retraining removes.
#
# Starts from the released adapter by default (INIT_ADAPTER). No vLLM needed.
#
#   bash scripts/train_iterret_osam.sh
#   GPU=0 EPOCHS=3 LR=5e-5 bash scripts/train_iterret_osam.sh
#   MAX_WRITE_TOKENS=6144 bash scripts/train_iterret_osam.sh   # A100-80GB: near-full evidence
#
# Memory on a 24 GB 4090: 1536 write tokens OOM'd (22.5 GB); use 1024.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../env.sh"
caimms_activate

MAX_WRITE_TOKENS="${MAX_WRITE_TOKENS:-1536}"
MAX_READ_TOKENS="${MAX_READ_TOKENS:-512}"
TRAIN_FILE="${TRAIN_FILE:-${CAIMMS_OUTPUT_DIR}/sft_iterret_qasper_train_w${MAX_WRITE_TOKENS}.jsonl}"
INIT_ADAPTER="${INIT_ADAPTER:-${CAIMMS_WORKSPACE}/models/delta-mem-adapter}"
OUT_DIR="${OUT_DIR:-${CAIMMS_WORKSPACE}/models/delta-mem-iterret-w${MAX_WRITE_TOKENS}}"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NO_STATE_P="${NO_STATE_P:-0.2}"
STATE_ONLY_P="${STATE_ONLY_P:-0.2}"

[ -s "${TRAIN_FILE}" ] || { echo "MISSING training file ${TRAIN_FILE} -- run scripts/build_sft_data.sh"; exit 1; }
[ -s "${INIT_ADAPTER}/delta_mem_adapter.pt" ] || { echo "MISSING init adapter ${INIT_ADAPTER}"; exit 1; }

if python3 -c "import flash_attn" 2>/dev/null; then ATTN=flash_attention_2; else ATTN=sdpa; fi
ATTN="${ATTN_IMPL:-${ATTN}}"

echo "=============================================="
echo "  train file : ${TRAIN_FILE} ($(wc -l < "${TRAIN_FILE}") episodes)"
echo "  init       : ${INIT_ADAPTER}"
echo "  out        : ${OUT_DIR}"
echo "  write/read : ${MAX_WRITE_TOKENS}/${MAX_READ_TOKENS} tokens | attn ${ATTN} | GPU ${GPU}"
echo "  epochs ${EPOCHS}  lr ${LR}  grad-accum ${GRAD_ACCUM}  mix no_state=${NO_STATE_P} state_only=${STATE_ONLY_P}"
echo "=============================================="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

mkdir -p "${OUT_DIR}"
cd "${CAIMMS_ROOT}/delta-Mem"
# Less fragmentation: a 1536-token run OOM'd at 22.5 GB with ~570 MB reserved-but-unused.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${GPU}" python3 -u -m deltamem.train.delta_sft_experimental \
    --model-path "${CAIMMS_MODEL_PATH}" \
    --train-file "${TRAIN_FILE}" \
    --output-dir "${OUT_DIR}" \
    --init-adapter-dir "${INIT_ADAPTER}" \
    --no-tokenized-cache \
    --attn-implementation "${ATTN}" \
    --training-mode episode \
    --episode-recent-messages 1 \
    --assistant-loss-mode final_assistant_only \
    --max-write-length "${MAX_WRITE_TOKENS}" \
    --max-length "${MAX_READ_TOKENS}" \
    --memory-loss-mode context_ablation_ce \
    --context-ablation-mode mixed \
    --context-ablation-no-state-prob "${NO_STATE_P}" \
    --context-ablation-state-only-prob "${STATE_ONLY_P}" \
    --memory-write-granularity token \
    --learning-rate "${LR}" \
    --num-train-epochs "${EPOCHS}" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --warmup-ratio 0.05 \
    --bf16 \
    --logging-steps 10 \
    --save-steps 500 \
    --dataloader-num-workers 2 \
    2>&1 | tee "${OUT_DIR}/train.log"

echo "Adapter saved to ${OUT_DIR}. Evaluate with:"
echo "  CAIMMS_ADAPTER_DIR=${OUT_DIR} TAG=iterret_w${MAX_WRITE_TOKENS} ARMS=combined \\"
echo "    LB_MAX_EVIDENCE_TOKENS=${MAX_WRITE_TOKENS} bash scripts/run_longbench.sh"
