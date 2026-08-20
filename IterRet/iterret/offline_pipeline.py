
from __future__ import annotations

import sys
from typing import Any, Dict, List

from . import evaluator, learner
from .ctc_graph import CueTagContentGraph
from .experience_bank import ExperienceBank, build_default_embedding_backend, empty_experience_bank
from .graph import build_graph
from .llm_client import LLMClient
from .state import DEFAULT_MAX_ITERATIONS, new_state


def collect_trajectories(
    seed_questions: List[str],
    graph: CueTagContentGraph,
    llm: LLMClient,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> List[Dict[str, Any]]:
    unguided_bank = empty_experience_bank()
    compiled = build_graph(llm, graph, unguided_bank)

    records: List[Dict[str, Any]] = []
    for question in seed_questions:
        initial_state = new_state(question, max_iterations=max_iterations)
        try:
            final_state = compiled.invoke(initial_state, config={"recursion_limit": 25})
        except Exception as exc:  # noqa: BLE001 -- one bad bootstrap question shouldn't lose the rest
            print(f"[offline] warning: bootstrap trajectory for {question!r} failed, skipping it: {exc}",
                  file=sys.stderr)
            continue
        records.append({
            "original_query": question,
            "final_evidence": list(final_state.get("accumulated_evidence", [])),
            "steps": list(final_state.get("search_trajectory", [])),
        })
    return records


def construct_experience_banks(
    trajectory_records: List[Dict[str, Any]],
    llm: LLMClient,
    *,
    k_low: int = evaluator.K_LOW,
    k_high: int = evaluator.K_HIGH,
) -> ExperienceBank:
    bank = ExperienceBank(build_default_embedding_backend())

    for record in trajectory_records:
        original_query = record["original_query"]
        final_evidence = record["final_evidence"]

        for step in record["steps"]:
            if step["decision"] == "answer":
                continue  # rubrics target Planning/Reflection steps only (R2-Mem Sec. 4.2)

            score, reason, advice = evaluator.score_step(step, llm)
            quality = evaluator.classify(score, k_low=k_low, k_high=k_high)
            if quality == "discard":
                continue

            for module in ("Planning", "Reflection"):
                entry = learner.distill_experience(
                    step, reason, advice, quality, module, llm,
                    original_query=original_query, accumulated_evidence=final_evidence,
                )
                bank.add_experience(entry["condition"], entry["situation"], entry["experience"], module)

    return bank
