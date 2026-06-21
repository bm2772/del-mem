"""
FluxMem Memory Structures
Implements linear, graph, and hierarchical memory organization for MTEM episodic units.
Based on: "Choosing How to Remember: Adaptive Memory Structures for LLM Agents"
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
import numpy as np


# ---------------------------------------------------------------------------
# Core data primitives
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """A single user-agent exchange (one turn)."""
    page_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_text: str = ""
    agent_text: str = ""
    timestamp: float = field(default_factory=time.time)
    embedding: np.ndarray | None = None          # dense vector, set externally
    last_access: float = field(default_factory=time.time)

    def touch(self):
        self.last_access = time.time()

    def to_text(self) -> str:
        return f"User: {self.user_text}\nAgent: {self.agent_text}"


@dataclass
class EpisodicSession:
    """
    One episodic memory unit inside MTEM.
    Holds a group of semantically / temporally related Pages plus
    whichever indexing structure was selected for this unit.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pages: list[Page] = field(default_factory=list)
    summary: str = ""
    summary_embedding: np.ndarray | None = None
    structure_type: str = "linear"              # "linear" | "graph" | "hierarchical"
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    # structure-specific indices (populated by MemoryStructure classes)
    graph_index: dict[str, Any] = field(default_factory=dict)   # for graph memory
    topic_tree: dict[str, Any] = field(default_factory=dict)    # for hierarchical memory

    # utility tracking
    access_count: int = 0
    interaction_intensity: float = 0.0

    def touch(self):
        self.last_access = time.time()
        self.access_count += 1

    def utility_score(
        self,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
        now: float | None = None,
    ) -> float:
        """
        U(s) = w1*c(s) + w2*l(s) + w3*d(s)
        c = access frequency (normalised to [0,1] with log), l = intensity, d = recency
        """
        now = now or time.time()
        c = min(1.0, np.log1p(self.access_count) / 10.0)
        l_ = min(1.0, self.interaction_intensity)
        age = now - self.last_access
        d = np.exp(-age / (3600 * 24))             # decays over ~1 day
        return w1 * c + w2 * l_ + w3 * d

    def to_text(self) -> str:
        parts = [self.summary] if self.summary else []
        for p in self.pages:
            parts.append(p.to_text())
        return "\n---\n".join(parts)


@dataclass
class LTSMEntry:
    """One consolidated entry in Long-Term Semantic Memory."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    entry_type: str = "fact"                # "fact" | "profile" | "procedural"
    embedding: np.ndarray | None = None
    usage_count: int = 0
    last_access: float = field(default_factory=time.time)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def touch(self):
        self.last_access = time.time()
        self.usage_count += 1

    def is_eligible(
        self,
        tau_u: float = 1,
        tau_r: float = 0.01,
        tau_c: float = 0.0,
        now: float | None = None,
    ) -> bool:
        """Eq. 6 – keep entry iff all thresholds are met."""
        now = now or time.time()
        age = now - self.last_access
        recency = np.exp(-age / (3600 * 24 * 7))   # weekly decay
        return (
            self.usage_count >= tau_u
            and recency >= tau_r
            and (tau_c == 0.0 or self.confidence >= tau_c)
        )


# ---------------------------------------------------------------------------
# Memory structure implementations
# ---------------------------------------------------------------------------

class LinearMemory:
    """
    Chronological sequence of pages.
    Retrieval: cosine similarity + implicit recency weight.
    """

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        scored: list[tuple[float, int, Page]] = []
        n = len(session.pages)
        for i, page in enumerate(session.pages):
            if page.embedding is None:
                sim = 0.0
            else:
                sim = float(_cosine(query_emb, page.embedding))
            recency_weight = (i + 1) / n           # later pages score higher
            score = 0.7 * sim + 0.3 * recency_weight
            scored.append((score, i, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, _, p in scored[:top_k]]


class GraphMemory:
    """
    Entity-relation graph over pages.
    Nodes  = pages;  edges = shared entity mentions / high embedding similarity.
    Retrieval: start from highest-similarity node, expand 1-hop neighbours.
    """

    def build_index(self, session: EpisodicSession) -> None:
        """Build adjacency list stored in session.graph_index."""
        pages = session.pages
        n = len(pages)
        adj: dict[str, list[str]] = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                if pages[i].embedding is not None and pages[j].embedding is not None:
                    sim = float(_cosine(pages[i].embedding, pages[j].embedding))
                    if sim > 0.6:                   # edge threshold
                        adj[pages[i].page_id].append(pages[j].page_id)
                        adj[pages[j].page_id].append(pages[i].page_id)

        session.graph_index = {
            "adj": dict(adj),
            "id_to_idx": {p.page_id: idx for idx, p in enumerate(pages)},
        }

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        if not session.graph_index:
            self.build_index(session)

        pages = session.pages
        adj = session.graph_index.get("adj", {})
        id_to_idx = session.graph_index.get("id_to_idx", {})

        # seed node: highest cosine similarity to query
        sims = []
        for p in pages:
            s = float(_cosine(query_emb, p.embedding)) if p.embedding is not None else 0.0
            sims.append(s)
        seed_idx = int(np.argmax(sims))
        seed_page = pages[seed_idx]

        # collect seed + 1-hop neighbours
        neighbour_ids = adj.get(seed_page.page_id, [])
        candidate_ids = {seed_page.page_id} | set(neighbour_ids)

        candidate_pages = [pages[id_to_idx[pid]] for pid in candidate_ids if pid in id_to_idx]

        # re-rank by similarity
        scored = []
        for p in candidate_pages:
            s = float(_cosine(query_emb, p.embedding)) if p.embedding is not None else 0.0
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]


class HierarchicalMemory:
    """
    Topic tree over pages.
    Retrieval: coarse-to-fine DFS – match topic cluster, then pages within.
    """

    def build_index(self, session: EpisodicSession) -> None:
        """
        Lightweight clustering: group pages by rough embedding similarity
        using a greedy single-pass approach (no external dependencies).
        Stores cluster centres + member page ids in session.topic_tree.
        """
        pages = session.pages
        if not pages:
            session.topic_tree = {"clusters": []}
            return

        clusters: list[dict[str, Any]] = []
        threshold = 0.5

        for page in pages:
            if page.embedding is None:
                continue
            placed = False
            for cluster in clusters:
                centre = cluster["centre"]
                if float(_cosine(page.embedding, centre)) >= threshold:
                    cluster["members"].append(page.page_id)
                    # update running mean
                    n = len(cluster["members"])
                    cluster["centre"] = (centre * (n - 1) + page.embedding) / n
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "centre": page.embedding.copy(),
                    "members": [page.page_id],
                })

        session.topic_tree = {
            "clusters": clusters,
            "id_to_idx": {p.page_id: idx for idx, p in enumerate(pages)},
        }

    def retrieve(
        self,
        session: EpisodicSession,
        query_emb: np.ndarray,
        top_k: int = 3,
    ) -> list[Page]:
        if not session.pages:
            return []

        if not session.topic_tree:
            self.build_index(session)

        pages = session.pages
        id_to_idx = session.topic_tree.get("id_to_idx", {})
        clusters = session.topic_tree.get("clusters", [])

        if not clusters:
            return pages[:top_k]

        # coarse step: rank clusters by centre similarity
        cluster_scores = [
            float(_cosine(query_emb, c["centre"])) if c["centre"] is not None else 0.0
            for c in clusters
        ]
        best_cluster_idx = int(np.argmax(cluster_scores))
        best_cluster = clusters[best_cluster_idx]

        # fine step: rank pages within best cluster
        member_pages = [
            pages[id_to_idx[pid]]
            for pid in best_cluster["members"]
            if pid in id_to_idx
        ]
        scored = []
        for p in member_pages:
            s = float(_cosine(query_emb, p.embedding)) if p.embedding is not None else 0.0
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [p for _, p in scored[:top_k]]

        # if we need more, pull from other clusters (DFS)
        if len(result) < top_k:
            remaining_order = sorted(
                range(len(clusters)),
                key=lambda i: cluster_scores[i],
                reverse=True,
            )
            for ci in remaining_order:
                if ci == best_cluster_idx:
                    continue
                for pid in clusters[ci]["members"]:
                    if len(result) >= top_k:
                        break
                    if pid in id_to_idx:
                        result.append(pages[id_to_idx[pid]])

        return result[:top_k]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity."""
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
