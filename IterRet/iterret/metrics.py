
from __future__ import annotations

import re
from typing import Tuple

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> list:
    text = _PUNCT_RE.sub(" ", str(text).lower())
    return text.split()


def token_f1(prediction: str, gold: str) -> float:
    return precision_recall_f1(prediction, gold)[2]


def precision_recall_f1(prediction: str, gold: str) -> Tuple[float, float, float]:
    pred_tokens = normalize_answer(prediction)
    gold_tokens = normalize_answer(gold)
    if not pred_tokens or not gold_tokens:
        f1 = 1.0 if not pred_tokens and not gold_tokens else 0.0
        return f1, f1, f1

    pred_counts: dict = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if pred_counts.get(tok, 0) > 0:
            overlap += 1
            pred_counts[tok] -= 1

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
