#!/usr/bin/env bash
# Test-plan A2 (LongMemEval, 1 history x 20 q): IterRet evidence + recent window
# (combined_recent) vs IterRet evidence only (combined), same code. Run under tmux:
#
#   tmux new -d -s qa2 "bash scripts/queue_a2.sh 2>&1 | tee -a ../outputs/queue_A2.log"
#
# Earlier attempts died when other users grabbed memory between our free-memory
# check and vLLM's allocation, so the card is RESERVED by scripts/gpu_hold.py the
# moment it is free (>= 22 GB, no processes) and handed over to vLLM and then the
# eval by run_pipeline.sh. Each arm is retried (resume keeps finished questions)
# until it has all 20 answers, at most 6 tries.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."
source env.sh
caimms_activate
O="${CAIMMS_OUTPUT_DIR}"

free_gpu() {   # first GPU with >= 22 GB free and no processes
    while true; do
        for g in 0 1; do
            f="$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
            n="$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader | wc -l)"
            [ "$f" -ge 22000 ] && [ "$n" -eq 0 ] && { echo $g; return; }
        done
        sleep 30
    done
}
answered() {
    [ -s "$1" ] || { echo 0; return; }
    python3 -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('prediction','').strip()))" "$1"
}

for ARM in combined_recent combined; do
    OUT="${O}/mab_longmemeval_sstar_${ARM}_released_A2_sessions_c1_q20.jsonl"
    for try in 1 2 3 4 5 6; do
        [ "$(answered "${OUT}")" -ge 20 ] && break
        G=$(free_gpu)
        READY="${O}/.gpu_hold_ready_$$"; rm -f "${READY}"
        python3 scripts/gpu_hold.py --gpu "$G" --keep-mib 10000 --ready-file "${READY}" --max-hold-min 30 &
        HP=$!
        while kill -0 "$HP" 2>/dev/null && [ ! -f "${READY}" ]; do sleep 1; done
        if [ ! -f "${READY}" ]; then echo "[queue] could not reserve GPU $G -- retrying"; sleep 60; continue; fi
        echo "[queue] A2 ${ARM} try ${try} on GPU ${G}, reserved by pid ${HP} ($(date))"
        CAIMMS_GPU_HOLDER_PID=$HP VLLM_GPU=$G EVAL_GPU=$G VLLM_PORT=8002 MAB_MAX_CONTEXTS=1 MAB_MAX_QUESTIONS=20 \
            TAG=released_A2 ARMS=$ARM bash scripts/run_mab.sh
        kill "$HP" 2>/dev/null; wait "$HP" 2>/dev/null
    done
    echo "[queue] A2 ${ARM}: $(answered "${OUT}")/20 answered"
done
echo "QUEUE A2 DONE $(date)"
