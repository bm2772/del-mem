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
) -> List[str]:
    """Run retrieve/reflect/route for up to max_iterations rounds.
    Returns deduplicated accumulated_evidence WITHOUT calling answer_node.

    Stopping conditions (priority order):
      1. Token budget exceeded — prevents vLLM 400 errors.
      2. Two consecutive iterations with zero new evidence — graph exhausted.
      3. route_after_reflect returns "answer".
      4. max_iterations reached.
    """
    state = new_state(question, max_iterations=max_iterations)

    seen: set[str] = set()
    deduped: List[str] = []
    consecutive_empty = 0

    for _ in range(max_iterations):
        if _approx_tokens(deduped) > _TOKEN_BUDGET:
            break

        state = retrieve_node(state, graph, bank, llm)
        state = reflect_node(state, graph, bank, llm)

        added = 0
        for ev in state.get("accumulated_evidence", []):
            fp = _fingerprint(ev)
            if fp not in seen:
                seen.add(fp)
                deduped.append(ev)
                added += 1

        if added == 0:
            consecutive_empty += 1
            if consecutive_empty >= 99:
                break
        else:
            consecutive_empty = 0

        if route_after_reflect(state) == "answer":
            break

    return deduped
