
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, TypedDict

from .memory_builder import DialogueTurn

CATEGORY_NAMES: Dict[int, str] = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}
DEFAULT_EVAL_CATEGORIES: Tuple[int, ...] = (1, 2, 3, 4)  # exclude adversarial, matching MRAgent/MemR3


class LoCoMoQuestion(TypedDict):
    question: str
    answer: str
    category: int
    evidence: List[str]


class LoCoMoConversation(TypedDict):
    sample_id: str
    turns: List[DialogueTurn]
    questions: List[LoCoMoQuestion]


def load_raw_locomo(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _session_keys(conversation: dict) -> List[str]:
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
    # sort numerically by the session index rather than lexicographically
    # (lexicographic would put session_10 before session_2).
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _turn_text(turn: dict) -> str:
    text = turn.get("text", "")
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shared an image: {caption}]"
    return text


def conversation_to_turns(conversation: dict, *, max_turns: Optional[int] = None) -> List[DialogueTurn]:
    """Flatten all sessions (in order) into a single DialogueTurn list,
    stamping each turn with its session's date_time string."""
    turns: List[DialogueTurn] = []
    for session_key in _session_keys(conversation):
        time = conversation.get(f"{session_key}_date_time")
        for turn in conversation[session_key]:
            turns.append(DialogueTurn(speaker=turn.get("speaker", "Unknown"), text=_turn_text(turn), time=time))
            if max_turns is not None and len(turns) >= max_turns:
                return turns
    return turns


def parse_conversation(raw_conv: dict, *, max_turns: Optional[int] = None,
                        categories: Tuple[int, ...] = DEFAULT_EVAL_CATEGORIES,
                        max_questions: Optional[int] = None) -> LoCoMoConversation:
    questions: List[LoCoMoQuestion] = []
    for qa in raw_conv.get("qa", []):
        category = qa.get("category")
        if category not in categories:
            continue
        answer = qa.get("answer", qa.get("adversarial_answer"))
        if answer is None:
            continue
        questions.append(LoCoMoQuestion(
            question=qa["question"], answer=str(answer), category=category,
            evidence=list(qa.get("evidence", [])),
        ))
        if max_questions is not None and len(questions) >= max_questions:
            break

    return LoCoMoConversation(
        sample_id=raw_conv.get("sample_id", "unknown"),
        turns=conversation_to_turns(raw_conv["conversation"], max_turns=max_turns),
        questions=questions,
    )


def split_bootstrap_eval(
    raw_conversations: list, *, bootstrap_fraction: float = 0.1,
) -> Tuple[list, list]:
    """Deterministically split conversations: the first
    ``round(N * bootstrap_fraction)`` (at least 1) seed the experience bank
    (R2-Mem's "10% of it to accumulate experience", App. B.1); the rest are
    held out for evaluation. Order is preserved so re-runs are reproducible."""
    n = len(raw_conversations)
    n_bootstrap = max(1, round(n * bootstrap_fraction)) if bootstrap_fraction > 0 else 0
    n_bootstrap = min(n_bootstrap, n - 1) if n > 1 else n
    return raw_conversations[:n_bootstrap], raw_conversations[n_bootstrap:]
