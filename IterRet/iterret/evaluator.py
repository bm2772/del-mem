
from __future__ import annotations

import json
from typing import Literal, Tuple

from .json_utils import parse_json_object
from .llm_client import LLMClient
from .state import SearchStep

PLANNING_RUBRICS = [
    "Info Needs Coverage",
    "Info Needs Non-Redundancy",
    "Tool-Info Alignment",
    "Planning Efficiency",
]

REFLECTION_RUBRICS = [
    "Sufficiency Judgment Accuracy",
    "Minimal Sufficiency Recognition",
    "Follow-up Query Quality",
    "Answer Completeness Awareness",
]

K_LOW = 5
K_HIGH = 10

Quality = Literal["good", "bad", "discard"]

_EVALUATOR_SYSTEM_PROMPT = """rubric_evaluation
You are an expert evaluator for an AI memory deep search system (R2-Mem style).
Score the given step along its module's rubric dimensions (0-3 each), sum
into an overall score, and give a short reason and actionable advice.
Reply as JSON: {"score": int, "reason": str, "advice": str}.
"""


def _rubrics_for(module: str) -> list:
    return PLANNING_RUBRICS if module == "Planning" else REFLECTION_RUBRICS


def score_step(step: SearchStep, llm: LLMClient) -> Tuple[int, str, str]:
    rubrics = _rubrics_for(step["module"])
    user_prompt = json.dumps({
        "module": step["module"],
        "rubrics": rubrics,
        "query_used": step["query_used"],
        "action_taken": step["action_taken"],
        "found_summary": step["found_summary"],
        "decision": step["decision"],
    })
    raw = llm.chat(_EVALUATOR_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_object(raw)
    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    reason = str(parsed.get("reason", "") or ("unparseable evaluator reply" if not parsed else ""))
    advice = str(parsed.get("advice", ""))
    return score, reason, advice


def classify(score: int, *, k_low: int = K_LOW, k_high: int = K_HIGH) -> Quality:
    if score > k_high:
        return "good"
    if score < k_low:
        return "bad"
    return "discard"
