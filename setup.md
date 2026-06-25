# C-AIMMS Pipeline Cluster Environment

This document records the exact library versions required to run the C-AIMMS (IterRet + Delta-Mem) dual-GPU pipeline on the SLURM cluster.

## Required Environment
*   **Python:** 3.11
*   **vLLM:** `vllm==0.8.5` (Critical for IterRet Qwen server stability)
*   **Pydantic:** `pydantic==2.9.2` (Fixes FastAPI schema conflicts with vLLM 0.8.5)
*   **PyTorch:** `torch==2.6.x` (CUDA 12.x compatible)

## Flash Attention Installation
To avoid building from source on the A100 nodes, download the specific pre-compiled wheel for Python 3.11 and CUDA 12:

    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
