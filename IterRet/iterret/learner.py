
from __future__ import annotations

import json

from .experience_bank import ExperienceEntry, Module, planning_condition, reflection_condition
from .json_utils import parse_json_object
from .llm_client import LLMClient
from .state import SearchStep

_DISTILL_SYSTEM_PROMPT = """experience_distillation
You are an AI TRACE Strategist/Auditor (R2-Mem style).
Derive a GENERALIZABLE experience from this judged trajectory step.
The output experience must follow the form:
IF <abstract situation> THEN <strategy>.
Do not copy concrete surface facts from the trace; treat the diagnosed
evaluation reason as authoritative supervision.
Reply as JSON: {"thinking": str, "summary": str, "situation": str, "experience": str}.
"""


def distill_experience(
    step: SearchStep,
    reason: str,
    advice: str,
    quality: str,
    module: Module,
    llm: LLMClient,
    *,
    original_query: str,
    accumulated_evidence: list,
) -> ExperienceEntry:
    user_prompt = json.dumps({
        "module": module,
        "quality": quality,
        "query_used": step["query_used"],
        "action_taken": step["action_taken"],
        "found_summary": step["found_summary"],
        "decision": step["decision"],
        "diagnosed_reason": reason,
        "advice": advice,
    })
    raw = llm.chat(_DISTILL_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_object(raw)
    situation = str(parsed.get("situation") or "an ambiguous retrieval situation")
    experience = str(parsed.get("experience") or f"IF {situation} THEN proceed cautiously")

    condition = (
        planning_condition(step["query_used"])
        if module == "Planning"
        else reflection_condition(original_query, accumulated_evidence)
    )
    return ExperienceEntry(
        condition=condition,
        situation=situation,
        experience=experience,
        module=module,
        embedding=None,  # filled in by ExperienceBank.add_experience
    )
