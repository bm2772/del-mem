from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from iterret import evaluator, learner
from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import (
    ExperienceBank,
    build_default_embedding_backend,
    empty_experience_bank,
    planning_condition,
    reflection_condition,
)
from iterret.llm_client import LLMClient, MockLLMClient, OpenAICompatibleLLMClient
from iterret.memory_builder import build_ctc_graph_from_dialogue
from iterret.nodes import answer_node, reflect_node, retrieve_node, route_after_reflect
from iterret.sample_dialogue import (
    sample_dialogue_bootstrap_questions,
    sample_dialogue_eval_questions,
    sample_dialogue_turns,
)
from iterret.state import IterRetState, new_state


# ----------------------------------------------------------------------
# tiny printing helpers, purely for readability of the narration
# ----------------------------------------------------------------------
def _banner(title: str, char: str = "=") -> None:
    line = char * max(len(title) + 4, 78)
    print(f"\n{line}\n{char}{char} {title}\n{line}")


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrated walkthrough of the offline + online ITERRET pipeline.")
    parser.add_argument("--use-real-llm", action="store_true",
                         help="Use a real OpenAI-compatible server instead of the default mock. "
                              "Don't combine this with another job already saturating the same server.")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--question", default=None,
                         help="Override the online-stage question (default: first held-out sample question).")
    return parser.parse_args()


def make_llm(args: argparse.Namespace) -> LLMClient:
    """A FRESH client per stage, deliberately -- MockLLMClient's canned replies are
    stateful (self._call_count), tuned per-script-invocation (build_memory.py,
    build_experience_bank.py, and iterret_demo.py each instantiate their own client in
    real usage, i.e. separate OS processes with separate counters). Sharing one mock
    client across all three stages in this single combined script would carry that
    counter over between stages and change the mock's behavior versus real usage --
    so each stage below gets its own make_llm() call instead of one shared instance."""
    if args.use_real_llm:
        return OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
    return MockLLMClient()


# ======================================================================
# OFFLINE STAGE, PART 1: build the CTC memory graph
# ======================================================================
def offline_build_graph(args: argparse.Namespace) -> CueTagContentGraph:
    _banner("OFFLINE STAGE -- PART 1: build the CTC memory graph", "#")
    print("Runs ONCE, before any question is asked, over the ENTIRE dialogue transcript at")
    print("once (not incrementally, not per-question). MRAgent Eq. 6 / Appendix B.1.")

    turns = sample_dialogue_turns()
    _sub(f"Input: {len(turns)} raw dialogue turns (iterret.sample_dialogue)")
    for i, t in enumerate(turns, 1):
        print(f"  [{i:>2}] ({t.get('time')}) {t['speaker']}: {t['text']}")

    llm = make_llm(args)
    graph = build_ctc_graph_from_dialogue(turns, llm)

    _sub("Result -- episodic layer (one per turn: F_tag_LLM + F_cue_LLM, Eq. 6)")
    for cid, node in sorted(graph.contents.items()):
        if node.layer != "episodic":
            continue
        cues_here = sorted({c for c, _g in graph.content_to_cue_tag.get(cid, set())})
        print(f"  {cid:>4} | tag={node.tags} | cues={cues_here}")
        print(f"        | {node.text}")

    _sub("Result -- semantic layer (stable, entity-anchored facts, Eq. 20)")
    semantic_ids = [cid for cid, n in graph.contents.items() if n.layer == "semantic"]
    if not semantic_ids:
        print("  (none extracted)")
    for cid in sorted(semantic_ids):
        node = graph.contents[cid]
        cues_here = sorted({c for c, _g in graph.content_to_cue_tag.get(cid, set())})
        print(f"  {cid:>4} | tag={node.tags} | cue={cues_here} | {node.text}")

    _sub("Result -- topic layer (groups >=2 episodes sharing a theme, Eq. 21)")
    topic_ids = [cid for cid, n in graph.contents.items() if n.layer == "topic"]
    if not topic_ids:
        print("  (none formed)")
    for cid in sorted(topic_ids):
        node = graph.contents[cid]
        print(f"  {cid:>4} | links_to_episodes={node.topic_links} | {node.text}")

    _sub("Graph summary")
    print(f"  total cues:                  {len(graph.cues)}")
    print(f"  episodic content nodes:      {sum(1 for n in graph.contents.values() if n.layer == 'episodic')}")
    print(f"  semantic content nodes:      {sum(1 for n in graph.contents.values() if n.layer == 'semantic')}")
    print(f"  topic content nodes:         {sum(1 for n in graph.contents.values() if n.layer == 'topic')}")
    print(f"  total cue-tag-content links: {sum(len(v) for v in graph.cue_tag_to_content.values())}")
    print("\n>>> This graph is now FROZEN. Nothing past this point ever calls")
    print(">>> graph.add_content() / graph.add_cue() / graph.link() again.")
    return graph


# ======================================================================
# OFFLINE STAGE, PART 2: build the experience bank
# ======================================================================
def offline_build_experience_bank(graph: CueTagContentGraph, args: argparse.Namespace) -> ExperienceBank:
    _banner("OFFLINE STAGE -- PART 2: build the Planning/Reflection experience bank", "#")
    print("Still entirely offline. Runs bootstrap questions through the SAME retrieve/")
    print("reflect/route/answer loop used online below, but UNGUIDED (empty bank) --")
    print("purely to generate trajectories for the Evaluator+Learner to learn from")
    print("(R2-Mem Algorithm 1). The bank built here is what the ONLINE stage reads from.")

    bootstrap_questions = sample_dialogue_bootstrap_questions()
    _sub(f"Bootstrap questions used ({len(bootstrap_questions)}) -- kept DISJOINT from the "
         f"eval question below")
    for q in bootstrap_questions:
        print(f"  - {q}")

    unguided_bank = empty_experience_bank()
    llm = make_llm(args)

    steps_for_distillation: List[Tuple[Dict[str, Any], str, list]] = []
    for q_idx, question in enumerate(bootstrap_questions, 1):
        _sub(f"Unguided trajectory {q_idx}/{len(bootstrap_questions)}: {question!r}")
        state = new_state(question, max_iterations=args.max_iterations)
        for rnd in range(args.max_iterations):
            state = retrieve_node(state, graph, unguided_bank, llm)
            r_step = state["search_trajectory"][-1]
            print(f"    [round {rnd + 1}] RETRIEVE: action={r_step['action_taken']!r} -> {r_step['found_summary']}")

            state = reflect_node(state, graph, unguided_bank, llm)
            f_step = state["search_trajectory"][-1]
            print(f"    [round {rnd + 1}] REFLECT:  {f_step['found_summary']}")

            decision = route_after_reflect(state)
            print(f"    [round {rnd + 1}] ROUTER:   -> {decision}")
            if decision == "answer":
                break

        state = answer_node(state, llm)
        print(f"    final evidence ({len(state['accumulated_evidence'])} item(s)):")
        for ev in state["accumulated_evidence"]:
            print(f"      - {ev}")
        print(f"    answer: {state['final_answer']}")

        for step in state["search_trajectory"]:
            if step["decision"] != "answer":  # rubrics target Planning/Reflection steps only
                steps_for_distillation.append((step, question, list(state["accumulated_evidence"])))

    _sub(f"Rubric-guided Evaluator + self-Reflection Learner over {len(steps_for_distillation)} step(s)")
    bank = ExperienceBank(build_default_embedding_backend())
    for step, question, final_evidence in steps_for_distillation:
        score, reason, advice = evaluator.score_step(step, llm)
        quality = evaluator.classify(score)
        print(f"  iter={step['iteration']} module={step['module']:<10} score={score:>2} -> {quality:<7}"
              f" | reason: {reason[:70]}")
        if quality == "discard":
            continue
        for module in ("Planning", "Reflection"):
            entry = learner.distill_experience(
                step, reason, advice, quality, module, llm,
                original_query=question, accumulated_evidence=final_evidence,
            )
            bank.add_experience(entry["condition"], entry["situation"], entry["experience"], module)
            print(f"      -> distilled {module} experience: {entry['experience']}")

    _sub("Experience bank summary")
    print(f"  Planning entries:   {len(bank.planning_bank)}")
    print(f"  Reflection entries: {len(bank.reflection_bank)}")
    print("\n>>> This bank is now FROZEN. Nothing past this point ever calls")
    print(">>> bank.add_experience() again.")
    return bank


# ======================================================================
# ONLINE STAGE: answer one held-out question, round by round
# ======================================================================
def online_answer_question(
    graph: CueTagContentGraph, bank: ExperienceBank, question: str, args: argparse.Namespace,
) -> IterRetState:
    _banner("ONLINE STAGE: answer a HELD-OUT question, round by round", "#")
    print("The only part that runs 'at query time'. It only ever READS the graph and bank")
    print("built above (graph traversal + bank.retrieve()) -- never writes to either.")
    print(f"\nQuestion: {question!r}")
    print("(NOT one of the bootstrap questions above -- this checks the bank actually")
    print(" generalizes, instead of reusing whatever the unguided run already happened to do.)")

    llm = make_llm(args)
    state = new_state(question, max_iterations=args.max_iterations)

    for rnd in range(args.max_iterations):
        _sub(f"Round {rnd + 1}")
        query = state.get("current_refined_query") or state["original_query"]
        tags_preview = state["active_set"]["tags"][:5]
        tags_suffix = "..." if len(state["active_set"]["tags"]) > 5 else ""
        print(f"  query this round:   {query!r}")
        print(f"  active_set before:  cues={state['active_set']['cues']}")
        print(f"                      tags={tags_preview}{tags_suffix}")

        # NOTE: this preview calls bank.retrieve() directly (no LLM call -- just
        # embedding + cosine similarity) using the raw condition as a stand-in for the
        # LLM-abstracted "situation" retrieve_node will actually use internally. It is
        # therefore an approximation, clearly labeled as such, chosen specifically so
        # this walkthrough doesn't have to duplicate a real LLM call (abstract_situation)
        # just to print advice text.
        plan_cond = planning_condition(query)
        plan_preview = bank.retrieve(plan_cond, plan_cond, module="Planning", k=2)
        if plan_preview:
            print("  planning advice (approximate preview):")
            for entry in plan_preview:
                print(f"    - {entry['experience']}")
        else:
            print("  planning advice (approximate preview): <no Planning entries matched>")

        state = retrieve_node(state, graph, bank, llm)
        r_step = state["search_trajectory"][-1]
        print(f"  RETRIEVE -> action={r_step['action_taken']!r}")
        print(f"           -> {r_step['found_summary']}")
        for cid in state.get("_scratch_new_retrieval", [])[:5]:
            print(f"           -> new: {cid} = {graph.contents[cid].display_text()}")

        reflect_cond = reflection_condition(state["original_query"], state["accumulated_evidence"])
        reflect_preview = bank.retrieve(reflect_cond, reflect_cond, module="Reflection", k=2)
        if reflect_preview:
            print("  reflection advice (approximate preview):")
            for entry in reflect_preview:
                print(f"    - {entry['experience']}")

        state = reflect_node(state, graph, bank, llm)
        f_step = state["search_trajectory"][-1]
        print(f"  REFLECT  -> {f_step['found_summary']}")
        print(f"           -> evidence so far ({len(state['accumulated_evidence'])}):")
        for ev in state["accumulated_evidence"]:
            print(f"              - {ev}")
        print(f"           -> gaps remaining: {state['information_gaps']}")

        decision = route_after_reflect(state)
        print(f"  ROUTER   -> {decision}")
        if decision == "answer":
            break

    state = answer_node(state, llm)
    _sub("Final answer")
    print(f"  {state['final_answer']}")
    return state


# ======================================================================
def main() -> None:
    args = parse_args()
    print("LLM backend:", "REAL" if args.use_real_llm else "MOCK (default -- zero GPU/network load)")
    if args.use_real_llm:
        probe = OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
        print(f"  -> {probe.base_url} (model={probe.model})")

    graph = offline_build_graph(args)
    bank = offline_build_experience_bank(graph, args)

    question = args.question or sample_dialogue_eval_questions()[0]
    final_state = online_answer_question(graph, bank, question, args)

    _banner("SUMMARY: what was built OFFLINE (once) vs. what ran ONLINE (per question)", "*")
    n_episodic = sum(1 for n in graph.contents.values() if n.layer == "episodic")
    n_semantic = sum(1 for n in graph.contents.values() if n.layer == "semantic")
    n_topic = sum(1 for n in graph.contents.values() if n.layer == "topic")
    print("OFFLINE, built once, reused for every question:")
    print(f"  - CTC graph: {len(graph.cues)} cues, {n_episodic} episodic + {n_semantic} semantic"
          f" + {n_topic} topic content node(s)")
    print(f"  - Experience bank: {len(bank.planning_bank)} Planning + {len(bank.reflection_bank)} "
          f"Reflection entries")
    print("\nONLINE, ran fresh for this one question, read-only against the above:")
    print(f"  - Question: {question!r}")
    print(f"  - Iterations used: {final_state.get('iteration_count')} / {args.max_iterations}")
    print(f"  - Final answer: {final_state.get('final_answer')}")
    print("\nNeither the graph nor the bank were modified during the online stage. Verify")
    print("independently any time with:")
    print(r"  grep -n 'add_content\|add_cue\|\.link(\|add_experience' iterret/nodes.py")
    print("(should print nothing -- those mutators are only ever called from")
    print(" memory_builder.py and the offline-stage code above, never from nodes.py).")


if __name__ == "__main__":
    main()
