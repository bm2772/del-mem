"""
FluxMem – Three-Layer Memory Hierarchy
======================================
STIM  – Short-Term Interaction Memory  (§3.2, capacity-4 LRU buffer)
MTEM  – Mid-Term Episodic Memory       (§3.2, §3.3, BMM-gated fusion)
LTSM  – Long-Term Semantic Memory      (§3.2, eligibility-pruned store)
"""

from __future__ import annotations
import time
import numpy as np
from typing import Protocol, runtime_checkable
from feature_extractor import extract_features

from memory_structures import (
    Page,
    EpisodicSession,
    LTSMEntry,
    LinearMemory,
    GraphMemory,
    HierarchicalMemory,
    _cosine,
)
from bmm_gate import BMMGate


# ---------------------------------------------------------------------------
# Embedder protocol – plug in any encoder (sentence-transformers, Qwen, …)
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    def encode(self, text: str) -> np.ndarray:
        ...


class DummyEmbedder:
    """Fallback: random unit vectors (replace with real encoder)."""
    dim: int = 384

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Selector protocol – plug in trained MLP classifier
# ---------------------------------------------------------------------------

@runtime_checkable
class StructureSelector(Protocol):
    def predict(self, features: np.ndarray) -> str:
        """Return one of: 'linear', 'graph', 'hierarchical'."""
        ...


class DefaultSelector:
    """
    Fallback round-robin selector (replace with your trained MLP).
    In practice you should pass in the MLP you've already trained.
    """
    _cycle = ["linear", "graph", "hierarchical"]
    _idx = 0

    def predict(self, features: np.ndarray) -> str:
        choice = self._cycle[self.__class__._idx % 3]
        self.__class__._idx += 1
        return choice


# ---------------------------------------------------------------------------
# STIM – Short-Term Interaction Memory  (§3.2)
# ---------------------------------------------------------------------------

class STIM:
    """
    Fixed-capacity buffer of the most recent Pages.
    Eviction policy: LRU (Least Recently Used).
    Capacity = 4 (matching human working-memory limit, §3.2).
    """

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.pages: list[Page] = []

    def add(self, page: Page) -> list[Page]:
        """
        Add a page and return any pages evicted to MTEM.
        """
        self.pages.append(page)
        evicted: list[Page] = []

        if len(self.pages) > self.capacity:
            # evict oldest by last_access (LRU), Eq. 5
            overflow = len(self.pages) - self.capacity
            sorted_pages = sorted(self.pages, key=lambda p: p.last_access)
            evicted = sorted_pages[:overflow]
            evicted_ids = {p.page_id for p in evicted}
            self.pages = [p for p in self.pages if p.page_id not in evicted_ids]

        return evicted

    def get_all(self) -> list[Page]:
        for p in self.pages:
            p.touch()
        return list(self.pages)

    def clear(self) -> list[Page]:
        pages = list(self.pages)
        self.pages = []
        return pages


# ---------------------------------------------------------------------------
# MTEM – Mid-Term Episodic Memory  (§3.2, §3.3, §3.4, §3.5)
# ---------------------------------------------------------------------------

class MTEM:
    """
    Repository of EpisodicSessions.
    - Receives evicted pages from STIM
    - Uses BMM gate to decide merge vs. new session
    - Uses StructureSelector to pick memory organisation
    - Consolidates high-utility sessions into LTSM
    """

    def __init__(
        self,
        embedder: Embedder,
        selector: StructureSelector,
        max_sessions: int = 2000,
        bmm_tau: float = 0.6,
        bmm_min_keep: int = 1,
        consolidation_threshold: float = 0.7,   # utility score to promote to LTSM
    ):
        self.embedder = embedder
        self.selector = selector
        self.max_sessions = max_sessions
        self.bmm = BMMGate(tau=bmm_tau, min_keep=bmm_min_keep)
        self.consolidation_threshold = consolidation_threshold
        self.sessions: list[EpisodicSession] = []

        self._linear = LinearMemory()
        self._graph = GraphMemory()
        self._hierarchical = HierarchicalMemory()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _session_score(self, session: EpisodicSession, query_emb: np.ndarray) -> float:
        """Cosine between query and session summary embedding."""
        if session.summary_embedding is None:
            return 0.0
        return _cosine(query_emb, session.summary_embedding)

    def _rebuild_structure_index(self, session: EpisodicSession) -> None:
        """(Re)build structure-specific index after adding pages."""
        if session.structure_type == "graph":
            self._graph.build_index(session)
        elif session.structure_type == "hierarchical":
            self._hierarchical.build_index(session)
        # linear needs no explicit index

    def _summarise(self, pages: list[Page]) -> str:
        """Lightweight rule-based summary (replace with LLM call if desired)."""
        texts = [p.to_text() for p in pages]
        joined = "\n".join(texts)
        # truncate to first 300 chars as a placeholder summary
        return joined[:300].strip()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def ingest_pages(self, pages: list[Page]) -> list[EpisodicSession]:
        """
        Receive evicted pages from STIM and integrate them into sessions.
        Returns sessions that were modified or created.
        """
        touched: list[EpisodicSession] = []

        for page in pages:
            if page.embedding is None:
                page.embedding = self.embedder.encode(page.to_text())

            # ---- BMM-gated merge decision --------------------------
            if not self.sessions:
                session = self._new_session([page])
                self.sessions.append(session)
                touched.append(session)
                continue

            scores = [self._session_score(s, page.embedding) for s in self.sessions]
            best_idx = self.bmm.best(scores)

            if best_idx is None:
                # no compatible session → start new one
                session = self._new_session([page])
                self.sessions.append(session)
            else:
                session = self.sessions[best_idx]
                session.pages.append(page)
                session.interaction_intensity = min(1.0, session.interaction_intensity + 0.1)
                # update summary embedding
                session.summary = self._summarise(session.pages)
                session.summary_embedding = self.embedder.encode(session.summary)
                self._rebuild_structure_index(session)
                session.touch()

            touched.append(session)

        # capacity guard
        self._prune_if_needed()
        return touched

    def _new_session(self, pages: list[Page]) -> EpisodicSession:
        """Create a new episodic session and choose its structure."""
        session = EpisodicSession(pages=list(pages))
        session.summary = self._summarise(pages)
        session.summary_embedding = self.embedder.encode(session.summary)
        session.interaction_intensity = 0.1

        # select structure
        features = extract_features(pages)
        session.structure_type = self.selector.predict(features)
        self._rebuild_structure_index(session)
        return session

    def _prune_if_needed(self) -> list[EpisodicSession]:
        """Remove lowest-utility sessions when over capacity."""
        pruned = []
        while len(self.sessions) > self.max_sessions:
            utils = [s.utility_score() for s in self.sessions]
            worst_idx = int(np.argmin(utils))
            pruned.append(self.sessions.pop(worst_idx))
        return pruned

    def retrieve(
        self,
        query: str,
        query_emb: np.ndarray | None = None,
        top_k_sessions: int = 3,
        top_k_pages: int = 3,
    ) -> list[Page]:
        """
        Cross-session retrieval: rank sessions by summary similarity,
        then retrieve top pages from each using its assigned structure.
        """
        if not self.sessions:
            return []

        if query_emb is None:
            query_emb = self.embedder.encode(query)

        # rank sessions
        scored = sorted(
            self.sessions,
            key=lambda s: self._session_score(s, query_emb),
            reverse=True,
        )
        top_sessions = scored[:top_k_sessions]

        # retrieve pages from each session using its structure
        pages_out: list[Page] = []
        for session in top_sessions:
            session.touch()
            if session.structure_type == "graph":
                pages = self._graph.retrieve(session, query_emb, top_k=top_k_pages)
            elif session.structure_type == "hierarchical":
                pages = self._hierarchical.retrieve(session, query_emb, top_k=top_k_pages)
            else:
                pages = self._linear.retrieve(session, query_emb, top_k=top_k_pages)
            pages_out.extend(pages)

        # deduplicate by page_id (keep first occurrence)
        seen: set[str] = set()
        unique: list[Page] = []
        for p in pages_out:
            if p.page_id not in seen:
                seen.add(p.page_id)
                unique.append(p)
        return unique

    def sessions_ready_for_consolidation(self) -> list[EpisodicSession]:
        """Return high-utility sessions that should be promoted to LTSM."""
        return [s for s in self.sessions if s.utility_score() >= self.consolidation_threshold]


# ---------------------------------------------------------------------------
# LTSM – Long-Term Semantic Memory  (§3.2)
# ---------------------------------------------------------------------------

class LTSM:
    """
    Durable store of consolidated semantic knowledge (user facts, profiles,
    reusable strategies).  Entries are retained as long as they pass the
    eligibility criteria (Eq. 6).
    """

    def __init__(
        self,
        embedder: Embedder,
        max_entries: int = 100,
        tau_u: float = 1,       # minimum usage count
        tau_r: float = 0.01,    # minimum recency score
        tau_c: float = 0.0,     # minimum confidence (0 = disabled)
    ):
        self.embedder = embedder
        self.max_entries = max_entries
        self.tau_u = tau_u
        self.tau_r = tau_r
        self.tau_c = tau_c
        self.entries: list[LTSMEntry] = []

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def consolidate_session(self, session: EpisodicSession) -> list[LTSMEntry]:
        """
        Extract high-level facts from an episodic session and store them.
        In a full system, this calls an LLM summariser.  Here we use a
        lightweight heuristic that creates one fact per session.
        """
        new_entries: list[LTSMEntry] = []

        # one consolidated entry per session
        content = session.summary or session.to_text()[:300]
        emb = self.embedder.encode(content)

        # avoid duplicates: skip if very similar entry already exists
        for existing in self.entries:
            if existing.embedding is not None and _cosine(emb, existing.embedding) > 0.92:
                existing.touch()
                return new_entries         # already captured

        entry = LTSMEntry(
            content=content,
            entry_type="fact",
            embedding=emb,
        )
        self.entries.append(entry)
        new_entries.append(entry)
        self._prune_if_needed()
        return new_entries

    def add_entry(self, content: str, entry_type: str = "fact", confidence: float = 1.0) -> LTSMEntry:
        """Manually add a known fact / user profile item."""
        emb = self.embedder.encode(content)
        entry = LTSMEntry(content=content, entry_type=entry_type, embedding=emb, confidence=confidence)
        self.entries.append(entry)
        self._prune_if_needed()
        return entry

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_emb: np.ndarray | None = None,
        top_k: int = 5,
    ) -> list[LTSMEntry]:
        if not self.entries:
            return []

        if query_emb is None:
            query_emb = self.embedder.encode(query)

        scored = sorted(
            self.entries,
            key=lambda e: _cosine(query_emb, e.embedding) if e.embedding is not None else 0.0,
            reverse=True,
        )
        result = scored[:top_k]
        for e in result:
            e.touch()
        return result

    # ------------------------------------------------------------------
    # pruning (Eq. 6)
    # ------------------------------------------------------------------

    def _prune_if_needed(self) -> int:
        """Remove ineligible entries when over capacity."""
        removed = 0
        while len(self.entries) > self.max_entries:
            # remove the entry with lowest utility that also fails eligibility
            ineligible = [
                (i, e) for i, e in enumerate(self.entries)
                if not e.is_eligible(self.tau_u, self.tau_r, self.tau_c)
            ]
            if ineligible:
                worst_idx = ineligible[0][0]
            else:
                # all eligible → evict by lowest usage + recency
                scores = [
                    e.usage_count * 0.5 + np.exp(-(time.time() - e.last_access) / 86400) * 0.5
                    for e in self.entries
                ]
                worst_idx = int(np.argmin(scores))
            self.entries.pop(worst_idx)
            removed += 1
        return removed

    def prune_ineligible(self) -> int:
        """Explicit call to remove ineligible entries (maintenance)."""
        before = len(self.entries)
        self.entries = [
            e for e in self.entries
            if e.is_eligible(self.tau_u, self.tau_r, self.tau_c)
        ]
        return before - len(self.entries)
