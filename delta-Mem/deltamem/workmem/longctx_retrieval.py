"""Document graph + IterRet evidence, shared by the long-context eval
(eval_longbench_iterret) and the SFT-data builder (build_iterret_sft_data), so
the evidence delta-mem is TRAINED on is produced by exactly the code path it is
EVALUATED on.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Tuple

from iterret.ctc_graph import CueTagContentGraph
from iterret.doc_memory_builder import build_ctc_graph_from_document, chunk_document
from iterret.experience_bank import EmbeddingBackend, ExperienceBank, build_default_embedding_backend

from deltamem.workmem.evidence_filter import filter_evidence_by_relevance
from deltamem.workmem.iterret_bridge import get_iterret_evidence

ITERRET_MAX_ITERATIONS = 5


class LockedBackend(EmbeddingBackend):
    """Serialises encode() so one MiniLM model can serve several worker threads."""

    def __init__(self, inner: EmbeddingBackend) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def encode(self, text: str):
        with self._lock:
            return self._inner.encode(text)

    def similarity(self, a, b) -> float:
        return self._inner.similarity(a, b)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_backend(thread_safe: bool = False) -> EmbeddingBackend:
    backend = build_default_embedding_backend()
    if type(backend).__name__ == "KeywordOverlapEmbeddingBackend":
        # evidence_filter's cosine assumes float vectors; this backend returns
        # dicts and the filter then silently no-ops (upstream HANDOFF Sec. 6).
        raise RuntimeError("sentence-transformers MiniLM unavailable -- refusing to run "
                           "with the keyword fallback embedder")
    return LockedBackend(backend) if thread_safe else backend


def graph_cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def get_or_build_doc_graph(context: str, key: str, cache_dir: Path, llm, *,
                           segment_fn=None, segmentation: str = "fixed",
                           ) -> Tuple[CueTagContentGraph, List[str], bool]:
    """Returns (graph, passages in document order, was_cached).

    ``segment_fn(context) -> list[str]`` splits the document into episodic
    units; default = fixed ~180-word passages (``chunk_document``). The surprise
    segmenter is passed in here by the eval. Keep one cache dir per
    segmentation -- the cache is keyed by document only.
    """
    cache_path = graph_cache_path(cache_dir, key)
    if cache_path.exists():
        graph = CueTagContentGraph.load(str(cache_path))
        cached = True
    else:
        spans = segment_fn(context) if segment_fn is not None else chunk_document(context)
        graph = build_ctc_graph_from_document(spans, llm)
        graph.meta["segmentation"] = segmentation
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        graph.save(str(tmp))
        tmp.replace(cache_path)  # atomic: a killed run never leaves a half-written cache
        cached = False
    passages = [node.display_text() for node in graph.contents.values() if node.layer == "episodic"]
    return graph, passages, cached


def retrieve_evidence(question: str, graph: CueTagContentGraph, backend: EmbeddingBackend, llm,
                      diag: dict | None = None) -> List[str]:
    """IterRet retrieve/reflect/route (no answer node) + relevance sort -- the
    same two steps the LoCoMo eval applies before the OSAM write."""
    graph.attach_embedder(backend)
    bank = ExperienceBank(backend)  # empty bank, as in the LoCoMo runs
    diag = diag if diag is not None else {}
    evidence = get_iterret_evidence(question, graph, bank, llm,
                                    max_iterations=ITERRET_MAX_ITERATIONS, diag=diag)
    text_to_id = dict(zip(evidence, diag.get("evidence_ids", [])))
    if evidence:
        evidence = filter_evidence_by_relevance(question, evidence, backend.encode, threshold=0.30)
    # content id per returned item (the filter re-sorts, so map by text)
    diag["final_evidence_ids"] = [text_to_id.get(t, "?") for t in evidence]
    return evidence


# Default evidence presentation. On LongBench Qasper (50 docs) passages-only in
# paper order matched the whole paper (0.4623 vs vanilla 0.4581, 36/50 answers
# identical) while relevance order + LLM-extracted facts lost 0.10 (0.3554).
DEFAULT_EVIDENCE_LAYERS = "episodic"
DEFAULT_EVIDENCE_ORDER = "document"


def evidence_items(evidence: List[str], ids: List[str], graph: CueTagContentGraph) -> List[tuple]:
    """(text, content_id, layer, position-in-graph) per item, relevance order kept.
    Position = insertion order: passages in document order, then semantic facts
    in extraction order (extraction runs passage by passage)."""
    position = {cid: n for n, cid in enumerate(graph.contents)}
    out = []
    for text, cid in zip(evidence, ids):
        node = graph.contents.get(cid)
        out.append((text, cid, node.layer if node is not None else "?", position.get(cid, len(position))))
    return out


def present_evidence(items: List[tuple], *, layers: str = DEFAULT_EVIDENCE_LAYERS,
                     order: str = DEFAULT_EVIDENCE_ORDER, tokenizer=None,
                     max_tokens: int = 0) -> List[tuple]:
    """How retrieved evidence is shown to the model -- shared by the eval and by
    the SFT-episode builder so training and evaluation see the same format.

    items: relevance-sorted (text, id, layer, position) tuples (evidence_items).
    Steps, in this order:
      1. layers="episodic" drops semantic-fact nodes (LLM-extracted sentences);
         "all" keeps them.
      2. optional token cap (max_tokens > 0) keeps the MOST RELEVANT items.
      3. order="document" re-sorts the survivors by position in the document;
         "relevance" keeps retrieval order.
    """
    if layers == "episodic":
        items = [it for it in items if it[2] == "episodic"]
    elif layers != "all":
        raise ValueError(f"unknown evidence layers {layers!r}")
    if max_tokens > 0 and tokenizer is not None:
        # cap_evidence_by_tokens keeps a prefix (most relevant first)
        items = items[:len(cap_evidence_by_tokens([it[0] for it in items], tokenizer, max_tokens))]
    if order == "document":
        items = sorted(items, key=lambda it: it[3])
    elif order != "relevance":
        raise ValueError(f"unknown evidence order {order!r}")
    return items


def cap_evidence_by_tokens(evidence: List[str], tokenizer, max_tokens: int) -> List[str]:
    """Keep the most relevant evidence (list is relevance-sorted, best first)
    until ~max_tokens of chat-formatted write history. Always keeps >= 1 item.
    Used identically when building training episodes and (optionally) at eval
    time, so a retrained adapter sees the same evidence budget in both."""
    if max_tokens <= 0 or not evidence:
        return evidence
    kept, used = [], 0
    for item in evidence:
        n = len(tokenizer(item, add_special_tokens=False)["input_ids"]) + 6  # chat-template overhead
        if kept and used + n > max_tokens:
            break
        kept.append(item)
        used += n
    return kept
