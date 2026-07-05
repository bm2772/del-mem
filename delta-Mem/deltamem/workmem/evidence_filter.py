"""Filter accumulated_evidence by cosine similarity to the question.

Accepts any callable encode_fn(text: str) -> List[float] — works with
bank.backend.encode directly, no SentenceTransformer import needed here.

KNOWN LIMITATION: uses the same all-MiniLM-L6-v2 that did the retrieval.
Will NOT catch subtle semantic near-duplicates (sunrise vs sunset).
WILL catch zero-relevance garbage and very short filler strings.
Threshold 0.30 is intentionally loose — garbage filter only.
"""
from __future__ import annotations
from typing import Callable, List
import math

_MIN_WORDS = 3
_FALLBACK_N = 2


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def filter_evidence_by_relevance(
    question: str,
    evidence_list: List[str],
    encode_fn: Callable[[str], List[float]],
    threshold: float = 0.30,
) -> List[str]:
    """
    Args:
        question:      raw question string
        evidence_list: deduplicated output from get_iterret_evidence()
        encode_fn:     callable that maps a string to a float vector.
                       Pass bank.backend.encode directly.
        threshold:     cosine similarity floor. 0.30 keeps almost everything
                       except near-zero relevance garbage.
    Returns:
        Filtered list, always non-empty if input was non-empty.
    """
    if not evidence_list:
        return evidence_list
    try:
        long_enough = [e for e in evidence_list if len(e.split()) >= _MIN_WORDS]
        if not long_enough:
            return evidence_list

        q_vec = encode_fn(question)
        scored = [(e, _cosine(q_vec, encode_fn(e))) for e in long_enough]
        above = [e for e, s in scored if s >= threshold]

        if above:
            order = {e: i for i, e in enumerate(long_enough)}
            return sorted(above, key=lambda e: order.get(e, 999))
        else:
            # Everything below threshold: graph miss or embedding collapse.
            # Return top-N rather than empty list — empty list → score=0 skip.
            return [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:_FALLBACK_N]]
    except Exception as exc:
        print(f"[evidence_filter] Warning: failed ({exc}), returning unfiltered.")
        return evidence_list
