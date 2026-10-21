"""Hold a GPU's free memory until our own job is ready to take it over.

On a shared box, "the GPU is free" and "vLLM has allocated its memory" are
~40 s apart; twice another user's job landed in that gap and our vLLM died with
"No available memory for the cache blocks". This placeholder closes the gap:

  python3 scripts/gpu_hold.py --gpu 1 --keep-mib 10000 --ready-file /tmp/x &

  1. allocates (almost) all free memory on the GPU in 256 MiB blocks, writes
     --ready-file and holds;
  2. SIGUSR1 -> releases down to --keep-mib (room for vLLM to start; the rest
     stays reserved for the eval process that loads after vLLM is up);
  3. SIGTERM / SIGINT -> releases everything and exits.

run_pipeline.sh sends USR1 right before launching vLLM and TERM right before the
eval loads (when CAIMMS_GPU_HOLDER_PID is set). --max-hold-min bounds the hold so
a crashed queue can never squat on a shared GPU.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

BLOCK_MIB = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--keep-mib", type=int, default=10000, help="held after SIGUSR1")
    ap.add_argument("--min-mib", type=int, default=20000, help="exit 1 if less than this could be grabbed")
    ap.add_argument("--leave-mib", type=int, default=300, help="left free when grabbing")
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--max-hold-min", type=float, default=30.0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    torch.cuda.init()
    blocks = []
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free // 2**20 < BLOCK_MIB + args.leave_mib:
            break
        try:
            blocks.append(torch.empty(BLOCK_MIB * 2**20, dtype=torch.uint8, device="cuda"))
        except torch.cuda.OutOfMemoryError:
            break
    held = len(blocks) * BLOCK_MIB
    if held < args.min_mib:
        print(f"[gpu-hold] GPU {args.gpu}: only {held} MiB obtainable (need {args.min_mib}) -- giving up", flush=True)
        sys.exit(1)
    print(f"[gpu-hold] GPU {args.gpu}: holding {held} MiB (pid {os.getpid()})", flush=True)

    state = {"stop": False}

    def shrink(*_):
        keep = args.keep_mib // BLOCK_MIB
        del blocks[keep:]
        torch.cuda.empty_cache()
        print(f"[gpu-hold] GPU {args.gpu}: released down to {len(blocks) * BLOCK_MIB} MiB", flush=True)

    def stop(*_):
        state["stop"] = True

    signal.signal(signal.SIGUSR1, shrink)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with open(args.ready_file, "w") as fh:
        fh.write(str(os.getpid()))

    deadline = time.time() + args.max_hold_min * 60
    while not state["stop"] and time.time() < deadline:
        time.sleep(0.5)
    if not state["stop"]:
        print(f"[gpu-hold] GPU {args.gpu}: max hold {args.max_hold_min} min reached -- releasing", flush=True)
    blocks.clear()
    torch.cuda.empty_cache()
    print(f"[gpu-hold] GPU {args.gpu}: released all", flush=True)


if __name__ == "__main__":
    main()
