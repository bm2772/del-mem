
from __future__ import annotations

import json
import sys
from typing import Iterator, List, Optional, TypedDict

from .ctc_graph import CueTagContentGraph
from .json_utils import parse_json_object
from .llm_client import LLMClient

_EPISODE_EXTRACTION_SYSTEM_PROMPT = """episode_extraction
You build a Cue-Tag-Episode memory graph (MRAgent style) from one dialogue
turn. Read the turn and produce:
- "tag": a short phrase (<=4 words) summarizing the relational pattern of
  this episode (e.g. "Pet Adoption", "Job Change").
- "cues": a list of fine-grained cues -- entities, names, attributes, or
  salient descriptors explicitly mentioned in the turn (e.g. speaker name,
  named entities, key nouns). 2-6 cues.
Reply as JSON: {"tag": str, "cues": [str, ...]}.
"""

_SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """semantic_extraction
You extract stable, entity-anchored semantic facts (MRAgent Cue-Tag-Semantic
layer) from a dialogue transcript -- personal attributes, preferences, or
general facts that persist across episodes rather than describing a single
event. For each fact, identify the entity-level cue it is anchored to (e.g.
a person's name), an aspect-level tag (e.g. "Preference", "Occupation",
"Personality"), and the fact content as a short sentence.
Reply as JSON: {"semantics": [{"cue": str, "tag": str, "content": str}, ...]}.
If there are no stable facts beyond what episodes already capture, reply
with {"semantics": []}.
"""

_TOPIC_ABSTRACTION_SYSTEM_PROMPT = """topic_abstraction
You group episodic memory units (MRAgent Abstraction layer) into topic
nodes. Given a list of episodes (id, tag, text), identify recurring themes
shared across two or more episodes and summarize each as a topic.
Reply as JSON: {"topics": [{"topic": str, "episode_ids": [str, ...]}, ...]}.
Every episode_ids entry must be one of the given episode ids. Skip topics
that would only contain a single episode.
"""

DEFAULT_MAX_CHARS_PER_CALL = 4000  # conservative for an 8k-token context server


class DialogueTurn(TypedDict, total=False):
    speaker: str
    text: str
    time: Optional[str]


def _warn(message: str) -> None:
    print(f"[memory-builder] warning: {message}", file=sys.stderr)


def _safe_chat(system_prompt: str, user_prompt: str, llm: LLMClient, *, on_error: str) -> dict:
    """chat() + parse_json_object(), but a raised exception (e.g. the
    server rejecting an over-budget request) degrades to {} with a
    printed warning instead of aborting the whole graph build."""
    try:
        raw = llm.chat(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any backend, any failure mode
        _warn(f"{on_error}: {exc}")
        return {}
    return parse_json_object(raw)


def _iter_chunks(items: List[dict], *, text_key: str, max_chars: int) -> Iterator[List[dict]]:
    """Batch items so the joined length of item[text_key] (a proxy for
    token count -- ~4 chars/token for English) per batch stays under
    max_chars. A single item longer than max_chars still gets its own
    (oversized) batch rather than being silently dropped."""
    batch: List[dict] = []
    batch_len = 0
    for item in items:
        item_len = len(item[text_key])
        if batch and batch_len + item_len > max_chars:
            yield batch
            batch, batch_len = [], 0
        batch.append(item)
        batch_len += item_len
    if batch:
        yield batch


def _extract_episode(turn: DialogueTurn, llm: LLMClient) -> dict:
    parsed = _safe_chat(_EPISODE_EXTRACTION_SYSTEM_PROMPT, json.dumps(dict(turn)), llm,
                         on_error="episode extraction failed, using fallback tag/cues")
    tag = str(parsed.get("tag") or "Mention")
    cues = [str(c) for c in parsed.get("cues") or [] if str(c).strip()]
    if turn.get("speaker") and turn["speaker"] not in cues:
        cues.append(turn["speaker"])
    if not cues:
        cues = ["Unknown"]
    return {"tag": tag, "cues": cues}


def _extract_semantics(episode_summaries: List[dict], llm: LLMClient, *, max_chars: int) -> List[dict]:
    cleaned: List[dict] = []
    for chunk in _iter_chunks(episode_summaries, text_key="text", max_chars=max_chars):
        full_text = "\n".join(s["text"] for s in chunk)
        parsed = _safe_chat(_SEMANTIC_EXTRACTION_SYSTEM_PROMPT, full_text, llm,
                             on_error=f"semantic extraction skipped a {len(chunk)}-episode chunk")
        for item in parsed.get("semantics") or []:
            if not isinstance(item, dict):
                continue
            cue, tag, content = item.get("cue"), item.get("tag"), item.get("content")
            if cue and tag and content:
                cleaned.append({"cue": str(cue), "tag": str(tag), "content": str(content)})
    return cleaned


def _abstract_topics(episode_summaries: List[dict], llm: LLMClient, *, max_chars: int) -> List[dict]:
    if len(episode_summaries) < 2:
        return []
    valid_ids = {e["content_id"] for e in episode_summaries}
    topics: List[dict] = []
    for chunk in _iter_chunks(episode_summaries, text_key="text", max_chars=max_chars):
        if len(chunk) < 2:
            continue  # can't form a >=2-episode topic from a single-episode chunk
        parsed = _safe_chat(_TOPIC_ABSTRACTION_SYSTEM_PROMPT, json.dumps({"episodes": chunk}), llm,
                             on_error=f"topic abstraction skipped a {len(chunk)}-episode chunk")
        for item in parsed.get("topics") or []:
            if not isinstance(item, dict):
                continue
            topic_text = item.get("topic")
            episode_ids = [eid for eid in item.get("episode_ids") or [] if eid in valid_ids]
            if topic_text and len(episode_ids) >= 2:
                topics.append({"topic": str(topic_text), "episode_ids": episode_ids})
    return topics


def build_ctc_graph_from_dialogue(
    turns: List[DialogueTurn], llm: LLMClient, *, max_chars_per_call: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> CueTagContentGraph:
    """Run all three distillation stages and assemble the CTC graph.

    ``max_chars_per_call`` bounds how much episode text the semantic and
    topic stages pack into a single LLM call (see module docstring); lower
    it if your server's context window is smaller than ~8k tokens, raise
    it (carefully) if it's much larger and you want fewer, larger calls.
    """
    graph = CueTagContentGraph()
    episode_summaries: List[dict] = []

    # 1. Episodic layer: one (cue, tag, episode) set of links per turn.
    for i, turn in enumerate(turns):
        content_id = f"e{i + 1}"
        text = f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
        graph.add_content(content_id, text, layer="episodic", time=turn.get("time"))

        extracted = _extract_episode(turn, llm)
        for cue in extracted["cues"]:
            graph.link(cue, extracted["tag"], content_id)

        episode_summaries.append({"content_id": content_id, "tag": extracted["tag"], "text": text})

    # 2. Semantic layer: stable facts anchored to entity-level cues.
    for j, semantic in enumerate(_extract_semantics(episode_summaries, llm, max_chars=max_chars_per_call)):
        content_id = f"s{j + 1}"
        graph.add_content(content_id, semantic["content"], layer="semantic")
        graph.link(semantic["cue"], semantic["tag"], content_id)

    # 3. Abstraction layer: topic nodes wired to their constituent episodes.
    for k, topic in enumerate(_abstract_topics(episode_summaries, llm, max_chars=max_chars_per_call)):
        content_id = f"t{k + 1}"
        graph.add_content(content_id, f"Topic: {topic['topic']}", layer="topic",
                           topic_links=topic["episode_ids"])

    return graph
