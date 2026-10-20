# Source me. wait_for_gpu <gpu-index> <MiB-needed> [max-hours]
# Blocks until the GPU has at least <MiB-needed> free (polls every 2 min), so a
# job can be queued on a shared box and start by itself when the card frees up.
# Returns 1 after max-hours (default 24) without enough memory.
wait_for_gpu() {
    local gpu="$1" need="$2" max_h="${3:-24}" waited=0 free
    while true; do
        free="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [ "${free}" -ge "${need}" ]; then
            echo "[gpu-wait] GPU ${gpu}: ${free} MiB free -- starting ($(date))"
            return 0
        fi
        if [ "${waited}" -ge $((max_h * 3600)) ]; then
            echo "[gpu-wait] GPU ${gpu}: still only ${free} MiB free after ${max_h}h -- giving up"
            return 1
        fi
        [ $((waited % 1800)) -eq 0 ] && echo "[gpu-wait] GPU ${gpu}: ${free} MiB free, need ${need} -- waiting ($(date))"
        sleep 120
        waited=$((waited + 120))
    done
}
