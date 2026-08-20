"""Bridge: run ITERRET's real retrieve/reflect/route loop, stop BEFORE
answer_node, and return accumulated_evidence as a deduplicated list[str].

Changes vs original:
  - Deduplication: same graph node returned across iterations written once
    into OSAM, not N times. Prevents over-weighting of repeatedly-retrieved
    (often wrong) nodes in the S matrix.
  - Token budget guard: stops before accumulated evidence causes a vLLM 400
    error in the reflect_node prompt, which previously scored 0.
  - Empty-evidence early exit: two consecutive iterations returning nothing
    new means the graph has nothing left to offer — stop burning vLLM calls.

Note: route voting is NOT implemented. temperature=0.0 in llm_client.py makes
it degenerate — 3 identical calls, zero gain.
"""
from __future__ import annotations

import hashlib
from typing import List

from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import ExperienceBank
from iterret.llm_client import LLMClient
from iterret.nodes import reflect_node, retrieve_node, route_after_reflect
from iterret.state import DEFAULT_MAX_ITERATIONS, new_state

# Stop accumulating when approximate token count exceeds this.
# vLLM server limit is 8192; 6000 leaves ~2192 headroom for the reflect prompt.
_TOKEN_BUDGET = 6000


def _approx_tokens(text_list: List[str]) -> int:
    return int(sum(len(t.split()) for t in text_list) * 1.35)


def _fingerprint(text: str) -> str:
    normalised = " ".join(text.lower().split())[:200]
    return hashlib.md5(normalised.encode()).hexdigest()


def get_iterret_evidence(
    question: str,
    graph: CueTagContentGraph,
    bank: ExperienceBank,
    llm: LLMClient,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    diag: dict | None = None,
) -> List[str]:
    """Run retrieve/reflect/route for up to max_iterations rounds.
    Returns deduplicated accumulated_evidence WITHOUT calling answer_node.

    Stopping conditions (priority order):
      1. Token budget exceeded — prevents vLLM 400 errors.
      2. Two consecutive iterations with zero new evidence — graph exhausted.
      3. route_after_reflect returns "answer".
      4. max_iterations reached.

    ``diag``: optional dict, populated IN PLACE with per-question retrieval
    diagnostics (evidence content ids, how each round's routing decision was
    actually made, why the loop stopped). Passed as an out-parameter rather
    than folded into the return value so every existing caller keeps working
    unchanged -- the return type is still just list[str].
    """
    state = new_state(question, max_iterations=max_iterations)

    seen: set[str] = set()
    deduped: List[str] = []
    deduped_ids: List[str] = []
    consecutive_empty = 0
    rounds = 0
    stop_reason = "max_iterations"

    for _ in range(max_iterations):
        if _approx_tokens(deduped) > _TOKEN_BUDGET:
            stop_reason = "token_budget"
            break

        state = retrieve_node(state, graph, bank, llm)
        state = reflect_node(state, graph, bank, llm)
        rounds += 1

        evidence_texts = state.get("accumulated_evidence", [])
        # Index-aligned with evidence_texts by reflect_node. Guarded by
        # position rather than assumed, so a desync degrades to "?" instead
        # of raising and killing this question's retrieval outright.
        evidence_ids = state.get("accumulated_evidence_ids", [])

        added = 0
        for idx, ev in enumerate(evidence_texts):
            fp = _fingerprint(ev)
            if fp not in seen:
                seen.add(fp)
                deduped.append(ev)
                deduped_ids.append(evidence_ids[idx] if idx < len(evidence_ids) else "?")
                added += 1

        if added == 0:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                stop_reason = "graph_exhausted"
                break
        else:
            consecutive_empty = 0

        if route_after_reflect(state) == "answer":
            stop_reason = "route_answer"
            break

    if diag is not None:
        route_diagnostics = state.get("route_diagnostics", [])
        diag.update({
            "evidence_ids": deduped_ids,
            "rounds": rounds,
            "stop_reason": stop_reason,
            "action_parse_failures": state.get("action_parse_failures", 0),
            "route_modes": [d.get("route_mode") for d in route_diagnostics],
            # Longest raw routing reply seen this question. If this sits at
            # roughly the character equivalent of ROUTING_MAX_TOKENS on the
            # rounds that failed to parse, the cap is still binding and
            # should go up again; if parse failures happen at short lengths,
            # the cap was never the problem and the prompt is.
            "max_route_raw_len": max((d.get("raw_len", 0) for d in route_diagnostics), default=0),
        })

    return deduped
