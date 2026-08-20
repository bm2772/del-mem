from __future__ import annotations

import argparse
import json
import os

from iterret.ctc_graph import CueTagContentGraph
from iterret.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from iterret.offline_pipeline import collect_trajectories, construct_experience_banks
from iterret.state import DEFAULT_MAX_ITERATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ITERRET experience bank offline.")
    parser.add_argument("--use-real-llm", action="store_true",
                         help="Use a local OpenAI-compatible LLM server instead of the mock.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--llm-base-url", default=None,
                         help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1 "
                              "(default: $ITERRET_LLM_BASE_URL or http://localhost:8000/v1).")
    parser.add_argument("--llm-model", default=None,
                         help="Model name as reported by /v1/models, e.g. Qwen/Qwen3-4B-Instruct-2507 "
                              "(default: $ITERRET_LLM_MODEL or Qwen/Qwen3-4B-Instruct-2507).")
    parser.add_argument("--bootstrap-questions", default=None,
                         help="Path to a JSON list of bootstrap question strings. If omitted, "
                              "auto-selects the sample-dialogue questions when the loaded graph "
                              "looks like build_memory.py's sample dialogue (has a 'Maya' cue), "
                              "else falls back to demo_data's hand-seeded questions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    graph_path = os.path.join(args.data_dir, "ctc_graph.json")
    bank_path = os.path.join(args.data_dir, "experience_bank.json")

    if os.path.exists(graph_path):
        graph = CueTagContentGraph.load(graph_path)
        print(f"[offline] loaded CTC graph from {graph_path}")
    else:
        graph = seed_ctc_graph()
        graph.save(graph_path)
        print(f"[offline] seeded a new CTC graph and saved it to {graph_path}")

    llm = (OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
           if args.use_real_llm else MockLLMClient())
    if args.use_real_llm:
        print(f"[offline] using real LLM at {llm.base_url} (model={llm.model})")

    if args.bootstrap_questions:
        with open(args.bootstrap_questions, "r", encoding="utf-8") as fh:
            seed_questions = json.load(fh)
    elif "Maya" in graph.cues:
        seed_questions = sample_dialogue_bootstrap_questions()
    else:
        seed_questions = seed_bootstrap_questions()
    print(f"[offline] collecting trajectories for {len(seed_questions)} bootstrap question(s)...")
    trajectory_records = collect_trajectories(seed_questions, graph, llm, max_iterations=args.max_iterations)
    for record in trajectory_records:
        print(f"  - {record['original_query']!r}: {len(record['steps'])} step(s) logged")

    print("[offline] scoring steps and distilling experience...")
    bank = construct_experience_banks(trajectory_records, llm)
    bank.save(bank_path)
    print(f"[offline] saved {len(bank.planning_bank)} Planning + {len(bank.reflection_bank)} "
          f"Reflection experience entries to {bank_path}")


if __name__ == "__main__":
    main()
