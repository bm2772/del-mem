# C-AIMMS Pipeline Cluster Environment

This document records the environment required to run the C-AIMMS (IterRet + Delta-Mem)
dual-GPU pipeline. Verified on both a 2x A100 SLURM node (Mahamathi) and a 2x RTX 4090
workstation (resiliente-2003).

The authoritative pin list is `../requirements_exact.txt`. This file is the human-readable
summary of the parts that actually matter.

## Required Environment
*   **Python:** 3.11
*   **vLLM:** `vllm==0.8.5` (Critical for IterRet Qwen server stability)
*   **Pydantic:** `pydantic==2.13.4` -- this is what actually ran, on both machines,
    for every recorded result. An earlier version of this document claimed `2.9.2`
    was "critical for FastAPI schema conflicts with vLLM 0.8.5"; that claim has
    never been reproduced or isolated and no run has ever needed it. Treat it as
    unverified unless vLLM startup actually breaks.
*   **PyTorch:** `torch==2.6.x` (CUDA 12.x compatible)

## Flash Attention Installation
To avoid building from source on the A100 nodes, download the specific pre-compiled wheel for Python 3.11 and CUDA 12:

    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
