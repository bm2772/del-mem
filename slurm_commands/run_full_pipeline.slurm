#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=workmem-full
#SBATCH --partition=gpu
#SBATCH --output=workmem_pipeline_%j.log

source ~/.bashrc
conda activate workmem

# GPU 0: Background vLLM server
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model /home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager &

echo "Waiting for vLLM server to come online..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM server is ready!"

# GPU 1: Delta-Mem evaluation
cd /home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem
export PYTHONPATH="$(pwd):/home/kbasu/arnavbhatt/workmem_test/IterRet:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=1 python3 -m deltamem.workmem.eval_locomo_iterret_mock

echo "=============================================="
echo "  Pipeline finished at: $(date)"
echo "=============================================="
