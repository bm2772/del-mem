from __future__ import annotations
from typing import Callable, List
import math

_MIN_WORDS = 3


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def filter_evidence_by_relevance(
    question: str,
    evidence_list: List[str],
    encode_fn: Callable[[str], List[float]],
    threshold: float = 0.30,  # kept as parameter but not used for filtering
) -> List[str]:
    """Sort evidence by cosine similarity to question, highest first.

    No threshold filtering — the same embedding model that retrieved the
    evidence cannot reliably judge whether it is relevant (it retrieved it
    precisely because it thought it was relevant). Threshold filtering causes
    false drops on broad open-domain questions where similarity scores are
    uniformly low.

    What this does:
    - Drops strings under _MIN_WORDS (filler/noise)
    - Sorts remaining by similarity score descending so OSAM writes
      highest-confidence evidence into S first (delta rule is order-sensitive)
    - Never returns empty if input was non-empty
    """
    if not evidence_list:
        return evidence_list
    try:
        long_enough = [e for e in evidence_list if len(e.split()) >= _MIN_WORDS]
        if not long_enough:
            return evidence_list  # all strings too short; return unmodified
        q_vec = encode_fn(question)
        scored = [(e, _cosine(q_vec, encode_fn(e))) for e in long_enough]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [e for e, _ in ranked]
    except Exception as exc:
        print(f"[evidence_filter] Warning: failed ({exc}), returning unfiltered.")
        return evidence_list
