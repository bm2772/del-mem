from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from iterret import evaluator
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.graph import build_graph
from iterret.llm_client import LLMClient, MockLLMClient, OpenAICompatibleLLMClient
from iterret.llm_judge import judge_answer
from iterret.locomo_data import (
    CATEGORY_NAMES,
    DEFAULT_EVAL_CATEGORIES,
    LoCoMoConversation,
    load_raw_locomo,
    parse_conversation,
    split_bootstrap_eval,
)
from iterret.memory_builder import DEFAULT_MAX_CHARS_PER_CALL, build_ctc_graph_from_dialogue
from iterret.metrics import token_f1
from iterret.offline_pipeline import collect_trajectories, construct_experience_banks
from iterret.state import DEFAULT_MAX_ITERATIONS, new_state

_KNOWN_LOCOMO_PATHS = [
    "MRAgent/data/dataset_locomo.json",
    "Reflective-Experience-for-Memory-Search/data/locomo/locomo10.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory on a slice of LoCoMo, evaluate on the rest.")
    parser.add_argument("--locomo-path", default=None,
                         help="Path to LoCoMo's dataset_locomo.json / locomo10.json. "
                              f"Defaults to the first of {_KNOWN_LOCOMO_PATHS} that exists.")
    parser.add_argument("--use-real-llm", action="store_true",
                         help="Use a local OpenAI-compatible LLM server instead of the mock.")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.1,
                         help="Fraction of conversations used to build the offline memory/experience "
                              "bank; the rest are held out for evaluation (default 0.1, i.e. 10%%).")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--categories", default=",".join(str(c) for c in DEFAULT_EVAL_CATEGORIES),
                         help="Comma-separated LoCoMo category ids to evaluate "
                              "(1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial). "
                              f"Default excludes adversarial: {DEFAULT_EVAL_CATEGORIES}.")
    parser.add_argument("--max-turns-per-conversation", type=int, default=None,
                         help="Truncate each conversation's dialogue to this many turns before "
                              "building its CTC graph (default: no truncation).")
    parser.add_argument("--max-chars-per-call", type=int, default=DEFAULT_MAX_CHARS_PER_CALL,
                         help="Cap on episode-text characters packed into a single semantic/topic "
                              "extraction LLM call (default %(default)s). LoCoMo conversations run "
                              "hundreds of turns, so this matters a lot more here than for the small "
                              "sample dialogue -- lower it for smaller context-window servers.")
    parser.add_argument("--bootstrap-questions-per-conv", type=int, default=10,
                         help="How many of each bootstrap conversation's own questions to use for "
                              "unguided trajectory collection (default 10; keeps the offline stage fast).")
    parser.add_argument("--max-questions-per-conversation", type=int, default=None,
                         help="Cap on evaluated questions per held-out conversation (default: all).")
    parser.add_argument("--max-eval-conversations", type=int, default=None,
                         help="Cap on how many held-out conversations to evaluate (default: all).")
    parser.add_argument("--skip-llm-judge", action="store_true",
                         help="Skip the LLM-as-a-judge call and report token-F1 only.")
    parser.add_argument("--save-graphs-dir", default=None,
                         help="If set, save each conversation's CTC graph and the shared experience "
                              "bank as JSON under this directory for inspection/reuse.")
    parser.add_argument("--output", default=None, help="Path to dump full per-question results as JSON.")
    return parser.parse_args()


def resolve_locomo_path(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    for candidate in _KNOWN_LOCOMO_PATHS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a LoCoMo dataset file. Tried {_KNOWN_LOCOMO_PATHS}; pass --locomo-path explicitly."
    )


def build_offline_memory(
    bootstrap_raw: list, llm: LLMClient, *, max_turns: Optional[int], categories: tuple,
    bootstrap_questions_per_conv: int, max_iterations: int, max_chars_per_call: int,
) -> ExperienceBank:
    all_records: List[Dict[str, Any]] = []
    for raw_conv in bootstrap_raw:
        conv = parse_conversation(raw_conv, max_turns=max_turns, categories=categories,
                                   max_questions=bootstrap_questions_per_conv)
        print(f"[bootstrap] {conv['sample_id']}: distilling {len(conv['turns'])} turn(s) into a CTC graph...")
        graph = build_ctc_graph_from_dialogue(conv["turns"], llm, max_chars_per_call=max_chars_per_call)

        seed_questions = [q["question"] for q in conv["questions"]]
        print(f"[bootstrap] {conv['sample_id']}: collecting {len(seed_questions)} unguided trajectory(ies)...")
        records = collect_trajectories(seed_questions, graph, llm, max_iterations=max_iterations)
        all_records.extend(records)

    print(f"[bootstrap] scoring {len(all_records)} trajectory(ies) and distilling experience...")
    bank = construct_experience_banks(all_records, llm, k_low=evaluator.K_LOW, k_high=evaluator.K_HIGH)
    print(f"[bootstrap] built experience bank: {len(bank.planning_bank)} Planning + "
          f"{len(bank.reflection_bank)} Reflection entries")
    return bank


def evaluate_conversation(
    conv: LoCoMoConversation, graph, llm: LLMClient, bank: ExperienceBank, *,
    max_iterations: int, run_judge: bool,
) -> List[Dict[str, Any]]:
    compiled = build_graph(llm, graph, bank)

    results = []
    for qa in conv["questions"]:
        start = time.time()
        initial_state = new_state(qa["question"], max_iterations=max_iterations)
        try:
            final_state = compiled.invoke(initial_state, config={"recursion_limit": 8 * max_iterations + 5})
            predicted = final_state.get("final_answer") or ""
            iterations = final_state.get("iteration_count", 0)
        except Exception as exc:  # one bad question shouldn't abort the whole eval run
            predicted, iterations = f"[error: {exc}]", 0
        elapsed = time.time() - start

        f1 = token_f1(predicted, qa["answer"])
        correct: Optional[bool] = judge_answer(qa["question"], qa["answer"], predicted, llm) if run_judge else None

        results.append({
            "sample_id": conv["sample_id"],
            "category": qa["category"],
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "predicted_answer": predicted,
            "f1": f1,
            "correct": correct,
            "iterations": iterations,
            "elapsed_sec": elapsed,
        })
    return results


def print_results_table(results: List[Dict[str, Any]]) -> None:
    by_category: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def _avg(items: List[Dict[str, Any]], key: str) -> float:
        return sum(item[key] for item in items) / len(items) if items else 0.0

    have_judge = any(r["correct"] is not None for r in results)
    header = f"{'Category':<14}{'N':>6}{'F1':>9}" + (f"{'Judge(%)':>11}" if have_judge else "")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for category in sorted(by_category):
        items = by_category[category]
        name = CATEGORY_NAMES.get(category, str(category))
        line = f"{name:<14}{len(items):>6}{_avg(items, 'f1') * 100:>8.2f}%"
        if have_judge:
            judged = [i for i in items if i["correct"] is not None]
            judge_pct = 100 * sum(1 for i in judged if i["correct"]) / len(judged) if judged else 0.0
            line += f"{judge_pct:>10.2f}%"
        print(line)
    print("-" * len(header))
    overall_judge = ""
    if have_judge:
        judged_all = [r for r in results if r["correct"] is not None]
        overall_pct = 100 * sum(1 for r in judged_all if r["correct"]) / len(judged_all) if judged_all else 0.0
        overall_judge = f"{overall_pct:>10.2f}%"
    print(f"{'Overall':<14}{len(results):>6}{_avg(results, 'f1') * 100:>8.2f}%{overall_judge}")
    print("=" * len(header))

    avg_iter = _avg(results, "iterations")
    total_time = sum(r["elapsed_sec"] for r in results)
    print(f"\nAvg iterations/question: {avg_iter:.2f}  |  Total eval wall time: {total_time:.1f}s "
          f"over {len(results)} question(s)")


def main() -> None:
    args = parse_args()
    locomo_path = resolve_locomo_path(args.locomo_path)
    categories = tuple(int(c) for c in args.categories.split(",") if c.strip())

    llm = (OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
           if args.use_real_llm else MockLLMClient())
    if args.use_real_llm:
        print(f"[setup] using real LLM at {llm.base_url} (model={llm.model})")

    print(f"[setup] loading {locomo_path}")
    raw_conversations = load_raw_locomo(locomo_path)
    bootstrap_raw, eval_raw = split_bootstrap_eval(raw_conversations, bootstrap_fraction=args.bootstrap_fraction)
    if args.max_eval_conversations is not None:
        eval_raw = eval_raw[:args.max_eval_conversations]
    print(f"[setup] {len(raw_conversations)} conversation(s) total: "
          f"{len(bootstrap_raw)} for offline memory build, {len(eval_raw)} held out for evaluation")

    bank = build_offline_memory(
        bootstrap_raw, llm, max_turns=args.max_turns_per_conversation, categories=categories,
        bootstrap_questions_per_conv=args.bootstrap_questions_per_conv, max_iterations=args.max_iterations,
        max_chars_per_call=args.max_chars_per_call,
    )

    if args.save_graphs_dir:
        os.makedirs(args.save_graphs_dir, exist_ok=True)
        bank.save(os.path.join(args.save_graphs_dir, "experience_bank.json"))

    all_results: List[Dict[str, Any]] = []
    for raw_conv in eval_raw:
        conv = parse_conversation(raw_conv, max_turns=args.max_turns_per_conversation, categories=categories,
                                   max_questions=args.max_questions_per_conversation)
        print(f"\n[eval] {conv['sample_id']}: distilling {len(conv['turns'])} turn(s) into a CTC graph, "
              f"then answering {len(conv['questions'])} question(s)...")
        graph = build_ctc_graph_from_dialogue(conv["turns"], llm, max_chars_per_call=args.max_chars_per_call)
        if args.save_graphs_dir:
            graph.save(os.path.join(args.save_graphs_dir, f"{conv['sample_id']}_ctc_graph.json"))

        conv_results = evaluate_conversation(
            conv, graph, llm, bank, max_iterations=args.max_iterations, run_judge=not args.skip_llm_judge,
        )
        all_results.extend(conv_results)

        conv_f1 = sum(r["f1"] for r in conv_results) / len(conv_results) if conv_results else 0.0
        print(f"[eval] {conv['sample_id']}: done ({len(conv_results)} question(s), avg F1 {conv_f1 * 100:.2f}%)")

    print_results_table(all_results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "config": vars(args),
                "n_bootstrap_conversations": len(bootstrap_raw),
                "n_eval_conversations": len(eval_raw),
                "results": all_results,
            }, fh, indent=2)
        print(f"\n[output] wrote full results to {args.output}")


if __name__ == "__main__":
    main()
