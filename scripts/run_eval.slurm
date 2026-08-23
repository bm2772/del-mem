#!/bin/bash
#SBATCH --job-name=locomo_eval
#SBATCH --output=../outputs/locomo_eval_%j.log   # relative to submit dir (repo root) -> workspace outputs/
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # 8 CPU cores for fast data loading
#SBATCH --gres=gpu:1                   # Request exactly 1 GPU
#SBATCH --mem=64G                      # Request 64GB of CPU RAM
#SBATCH --time=24:00:00                # Request 24 hours of guaranteed runtime

echo "Running on node: $(hostname)"
echo "Starting evaluation at: $(date)"

# Activate the conda environment
source /home/kbasu/arnavbhatt/miniconda3/etc/profile.d/conda.sh
conda activate workmem

# Ensure the CUDA path we used earlier is still active
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Go to the delta-Mem directory
cd /home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem/

# Ensure output directory exists
mkdir -p outputs/

# Run the benchmark
python3 -m deltamem.eval.locomo_delta \
    --model-path /home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507 \
    --adapter-dir /home/kbasu/arnavbhatt/workmem_test/models/delta-mem-adapter \
    --device cuda:0 \
    --dtype bfloat16 \
    --eval-batch-size 8 \
    --output-json outputs/qwen3_delta_mem_locomo_eval.json

echo "Evaluation finished at: $(date)"
