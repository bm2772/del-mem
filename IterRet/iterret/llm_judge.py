
from __future__ import annotations

import json

from .json_utils import parse_json_object
from .llm_client import LLMClient

_JUDGE_SYSTEM_PROMPT = """answer_judge
You are grading whether a predicted answer correctly answers a question,
given a reference (gold) answer. Be lenient about paraphrasing, formatting,
units, and partial vs. full names/dates -- mark it correct if it conveys
the same factual content as the reference answer. Mark it incorrect if it
is missing, contradicts the reference, or says the information could not
be found while the reference shows it was answerable.
Reply as JSON: {"correct": true or false, "reason": str}.
"""


def judge_answer(question: str, gold_answer: str, predicted_answer: str, llm: LLMClient) -> bool:
    user_prompt = json.dumps({
        "question": question,
        "reference_answer": gold_answer,
        "predicted_answer": predicted_answer,
    })
    raw = llm.chat(_JUDGE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    parsed = parse_json_object(raw)
    verdict = parsed.get("correct")
    if isinstance(verdict, bool):
        return verdict
    return str(verdict).strip().lower() in ("true", "yes", "1", "correct")
