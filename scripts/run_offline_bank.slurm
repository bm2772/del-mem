#!/bin/bash
#SBATCH --job-name=caimms_offline
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/kbasu/arnavbhatt/workmem_test/outputs/offline_%j.log

set -euo pipefail

DELTA_MEM_ROOT="/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem"
ITERRET_ROOT="/home/kbasu/arnavbhatt/workmem_test/IterRet"
OUTPUT_DIR="/home/kbasu/arnavbhatt/workmem_test/outputs"
SERVER_LOG="${OUTPUT_DIR}/offline_server_${SLURM_JOB_ID}.log"

mkdir -p "${OUTPUT_DIR}"

source /home/kbasu/arnavbhatt/miniconda3/etc/profile.d/conda.sh
conda activate workmem
export PYTHONPATH="${DELTA_MEM_ROOT}:${ITERRET_ROOT}:${PYTHONPATH:-}"

echo "[1/3] Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507 \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --enforce-eager \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[2/3] Waiting for server..."
ELAPSED=0
until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 10; ELAPSED=$((ELAPSED+10))
    if [ $ELAPSED -ge 300 ]; then
        echo "ERROR: server timeout"; tail -20 "${SERVER_LOG}"
        kill $SERVER_PID; exit 1
    fi
done
echo "Server online."

echo "[3/3] Running offline bank construction..."
python3 -u << 'PYEOF'
import json, sys
sys.path.insert(0, "/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem")
sys.path.insert(0, "/home/kbasu/arnavbhatt/workmem_test/IterRet")

from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.memory_builder import build_ctc_graph_from_dialogue
from iterret.locomo_data import (
    load_raw_locomo, split_bootstrap_eval,
    parse_conversation, conversation_to_turns,
)
from iterret.offline_pipeline import collect_trajectories, construct_experience_banks

DATA_FILE   = "/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem/data/locomo10.json"
BANK_PATH   = "/home/kbasu/arnavbhatt/workmem_test/outputs/experience_bank.json"
VLLM_URL    = "http://localhost:8000/v1"
VLLM_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"

raw = load_raw_locomo(DATA_FILE)
bootstrap_raw, eval_raw = split_bootstrap_eval(raw, bootstrap_fraction=0.1)
print(f"Bootstrap conversations: {len(bootstrap_raw)}, Eval conversations: {len(eval_raw)}")

llm = OpenAICompatibleLLMClient(base_url=VLLM_URL, model=VLLM_MODEL, max_tokens=256)

all_trajectory_records = []

for i, raw_conv in enumerate(bootstrap_raw):
    print(f"\n[bootstrap conv {i}] Building CTC graph...")
    turns = conversation_to_turns(raw_conv["conversation"])
    graph = build_ctc_graph_from_dialogue(turns, llm)
    n_nodes = len(graph.nodes) if hasattr(graph, "nodes") else "?"
    print(f"[bootstrap conv {i}] Graph built: {n_nodes} nodes")

    parsed = parse_conversation(raw_conv)
    questions = [q["question"] for q in parsed["questions"]]
    print(f"[bootstrap conv {i}] Collecting trajectories for {len(questions)} questions...")

    records = collect_trajectories(questions, graph, llm, max_iterations=5)
    all_trajectory_records.extend(records)
    print(f"[bootstrap conv {i}] {len(records)} trajectories collected")

print(f"\nTotal trajectory records: {len(all_trajectory_records)}")
print("Scoring and distilling experience...")
bank = construct_experience_banks(all_trajectory_records, llm)

bank.save(BANK_PATH)
print(f"\nExperience bank saved to {BANK_PATH}")
print(f"Planning entries: {len(bank.planning_bank)}")
print(f"Reflection entries: {len(bank.reflection_bank)}")
PYEOF

kill $SERVER_PID 2>/dev/null || true
echo "Offline phase complete."
