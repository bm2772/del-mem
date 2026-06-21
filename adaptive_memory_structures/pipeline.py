"""
FluxMem Pipeline
================
Orchestrates STIM → MTEM → LTSM flow with Qwen3-4B as the backbone LLM.
Implements Eq. 3 & 4: memory-conditioned context assembly and response generation.

Usage
-----
    from pipeline import FluxMemPipeline

    pipeline = FluxMemPipeline(model_path="Qwen/Qwen3-4B")
    response = pipeline.chat(user_text="What did I say about my dog last week?")
    print(response)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memory_structures import Page, EpisodicSession, LTSMEntry
from memory_layers import STIM, MTEM, LTSM, DummyEmbedder, DefaultSelector, extract_features
from qwen_client import QwenClient, QwenConfig, ClientEmbedderAdaptor, ClientGeneratorAdaptor

logger = logging.getLogger(__name__)

# Re-export for any legacy imports
QwenEmbedder  = ClientEmbedderAdaptor
QwenGenerator = ClientGeneratorAdaptor


# ---------------------------------------------------------------------------
# Prompt builders  (Figure 7 / Appendix I)
# ---------------------------------------------------------------------------

def build_response_prompt(
    query: str,
    stim_pages: list[Page],
    mtem_pages: list[Page],
    ltsm_entries: list[LTSMEntry],
    user_a: str = "user_a",
    user_b: str = "agent",
) -> str:
    """
    Mirrors the paper's 'Prompt for Response' (Figure 7, Appendix I.1).
    """
    # recent context
    history_lines = []
    for p in stim_pages:
        history_lines.append(f"User: {p.user_text}")
        if p.agent_text:
            history_lines.append(f"Agent: {p.agent_text}")
    history_text = "\n".join(history_lines) if history_lines else "(none)"

    # retrieved episodic memories
    retrieval_lines = []
    for p in mtem_pages:
        retrieval_lines.append(f"- User: {p.user_text}")
        if p.agent_text:
            retrieval_lines.append(f"  Agent: {p.agent_text}")
    retrieval_text = "\n".join(retrieval_lines) if retrieval_lines else "(none)"

    # long-term facts / user profile
    profile_lines = [f"- {e.content}" for e in ltsm_entries]
    background = "\n".join(profile_lines) if profile_lines else "(none)"

    prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {user_a} and {user_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n"
        f"<User Profile>\n"
        f"Characteristics of {user_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {user_a} or {user_b} "
        f"in an extremely concise manner.\n"
    )
    return prompt


def build_meta_summary_prompt(last_meta: str, new_dialogue: str) -> str:
    """Mirrors Figure 8 – Prompt for Meta Info."""
    return (
        'Update the conversation meta-summary by incorporating the new dialogue '
        'while maintaining continuity.\n\n'
        'Guidelines:\n'
        '1. Start from the previous meta-summary (if exists)\n'
        '2. Add/update information based on the new dialogue\n'
        '3. Keep it concise (1-2 sentences max)\n'
        '4. Maintain context coherence\n\n'
        f'Previous Meta-summary: {last_meta}\n'
        f'New Dialogue:\n{new_dialogue}\n'
        'Updated Meta-summary:'
    )


# ---------------------------------------------------------------------------
# Retrieval fusion (dense + BM25 via reciprocal rank, §4.1.3)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[Page]],
    k: int = 60,
) -> list[Page]:
    """
    Combine multiple ranked lists via RRF.
    Returns pages sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    page_map: dict[str, Page] = {}

    for ranked in ranked_lists:
        for rank, page in enumerate(ranked, start=1):
            scores[page.page_id] = scores.get(page.page_id, 0.0) + 1.0 / (k + rank)
            page_map[page.page_id] = page

    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    return [page_map[pid] for pid in sorted_ids]


# ---------------------------------------------------------------------------
# FluxMem Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # memory layer sizes
    stim_capacity: int = 4
    mtem_max_sessions: int = 2000
    ltsm_max_entries: int = 100

    # BMM gate
    bmm_tau: float = 0.5            # τ_BMM — best from sensitivity analysis (§4.4)
    bmm_min_keep: int = 1           # m_min

    # LTSM eligibility
    ltsm_tau_u: float = 1
    ltsm_tau_r: float = 0.01
    ltsm_tau_c: float = 0.0

    # LTSM consolidation
    consolidation_utility_threshold: float = 0.7
    consolidation_every_n_turns: int = 10   # run consolidation pass every N turns

    # retrieval
    top_k_mtem_sessions: int = 3
    top_k_pages_per_session: int = 3
    top_k_ltsm: int = 5

    # generation
    max_new_tokens: int = 512
    temperature: float = 0.0

    # model
    model_path: str = "Qwen/Qwen3-4B"
    device: str = "auto"

    # users
    user_name: str = "user_a"
    agent_name: str = "agent"


class FluxMemPipeline:
    """
    End-to-end FluxMem conversational memory pipeline.

    Parameters
    ----------
    config : PipelineConfig
        All hyper-parameters.
    selector : StructureSelector | None
        Your pre-trained MLP selector.  Pass None to use the default
        round-robin (useful for testing before the selector is trained).
    embedder : Embedder | None
        External sentence encoder.  Pass None to use Qwen mean-pooling
        (slower but consistent) or DummyEmbedder for unit tests.
    model : Any, tokenizer : Any
        Pre-loaded HuggingFace model + tokenizer.  If both are None,
        the pipeline will load them from config.model_path.
    dry_run : bool
        If True, skip model loading and return placeholder responses.
        Useful for integration testing of the memory pipeline alone.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        selector=None,
        embedder=None,
        qwen: "QwenClient | None" = None,
        dry_run: bool = False,
    ):
        self.cfg = config or PipelineConfig()
        self.dry_run = dry_run
        self._turn_count = 0
        self._meta_summary = ""

        # ---- QwenClient (load once, share with rewards / judge) ------
        if dry_run:
            self._qwen = QwenClient.load(dry_run=True, registry_key="_pipeline_dry")
        elif qwen is not None:
            self._qwen = qwen
        else:
            qwen_cfg = QwenConfig(
                model_path=self.cfg.model_path,
                device=self.cfg.device,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
            )
            self._qwen = QwenClient.load(config=qwen_cfg)

        # ---- embedder -------------------------------------------------
        if embedder is not None:
            self._embedder = embedder
        elif dry_run:
            self._embedder = DummyEmbedder()
        else:
            self._embedder = self._qwen.as_embedder()

        # ---- generator (adaptor for backward-compat call sites) -------
        if dry_run:
            self._generator = None
        else:
            self._generator = self._qwen.as_generator()

        # ---- selector -------------------------------------------------
        self._selector = selector if selector is not None else DefaultSelector()

        # ---- memory layers --------------------------------------------
        self.stim = STIM(capacity=self.cfg.stim_capacity)
        self.mtem = MTEM(
            embedder=self._embedder,
            selector=self._selector,
            max_sessions=self.cfg.mtem_max_sessions,
            bmm_tau=self.cfg.bmm_tau,
            bmm_min_keep=self.cfg.bmm_min_keep,
            consolidation_threshold=self.cfg.consolidation_utility_threshold,
        )
        self.ltsm = LTSM(
            embedder=self._embedder,
            max_entries=self.cfg.ltsm_max_entries,
            tau_u=self.cfg.ltsm_tau_u,
            tau_r=self.cfg.ltsm_tau_r,
            tau_c=self.cfg.ltsm_tau_c,
        )

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def chat(self, user_text: str) -> str:
        """
        Process one user turn through the full FluxMem pipeline.

        Flow
        ----
        1. Embed query
        2. Retrieve from STIM (all), MTEM (structured), LTSM (semantic)
        3. Fuse retrieved context (RRF)
        4. Build prompt → generate response
        5. Write new page to STIM → evict to MTEM if needed
        6. Periodically consolidate MTEM → LTSM
        """
        self._turn_count += 1

        # ---- 1. embed query ------------------------------------------
        query_emb = self._embedder.encode(user_text)

        # ---- 2. retrieve from each layer -----------------------------
        stim_pages = self.stim.get_all()

        mtem_pages_dense = self.mtem.retrieve(
            query=user_text,
            query_emb=query_emb,
            top_k_sessions=self.cfg.top_k_mtem_sessions,
            top_k_pages=self.cfg.top_k_pages_per_session,
        )

        ltsm_entries = self.ltsm.retrieve(
            query=user_text,
            query_emb=query_emb,
            top_k=self.cfg.top_k_ltsm,
        )

        # ---- 3. fuse retrieved pages (RRF) ---------------------------
        # We treat stim_pages as the highest-priority ranked list
        fused_pages = reciprocal_rank_fusion(
            [stim_pages, mtem_pages_dense],
            k=60,
        )

        # ---- 4. build prompt & generate ------------------------------
        prompt = build_response_prompt(
            query=user_text,
            stim_pages=stim_pages,
            mtem_pages=fused_pages,
            ltsm_entries=ltsm_entries,
            user_a=self.cfg.user_name,
            user_b=self.cfg.agent_name,
        )

        if self.dry_run:
            response_text = f"[DRY-RUN] Would answer: '{user_text}' using {len(fused_pages)} memory pages."
        else:
            response_text = self._qwen.chat(prompt)

        # ---- 5. write new page to STIM -------------------------------
        new_page = Page(
            user_text=user_text,
            agent_text=response_text,
            timestamp=time.time(),
        )
        new_page.embedding = query_emb      # reuse already-computed query embedding
        evicted_pages = self.stim.add(new_page)

        # push evicted pages → MTEM
        if evicted_pages:
            self.mtem.ingest_pages(evicted_pages)

        # ---- 6. periodic consolidation MTEM → LTSM ------------------
        if self._turn_count % self.cfg.consolidation_every_n_turns == 0:
            self._consolidate()

        return response_text

    # ------------------------------------------------------------------
    # Consolidation pass
    # ------------------------------------------------------------------

    def _consolidate(self) -> None:
        """
        Promote high-utility MTEM sessions into LTSM (§3.2).
        Also prune stale LTSM entries.
        """
        candidates = self.mtem.sessions_ready_for_consolidation()
        promoted = 0
        for session in candidates:
            new_entries = self.ltsm.consolidate_session(session)
            promoted += len(new_entries)

        pruned = self.ltsm.prune_ineligible()
        logger.debug(
            f"[Consolidation turn={self._turn_count}] "
            f"promoted={promoted} entries, pruned={pruned} stale LTSM entries."
        )

    def force_consolidate(self) -> dict[str, int]:
        """Manually trigger a full consolidation pass. Returns stats."""
        candidates = self.mtem.sessions_ready_for_consolidation()
        promoted = sum(len(self.ltsm.consolidate_session(s)) for s in candidates)
        pruned = self.ltsm.prune_ineligible()
        return {"sessions_evaluated": len(candidates), "promoted": promoted, "pruned": pruned}

    # ------------------------------------------------------------------
    # Memory state inspection
    # ------------------------------------------------------------------

    def memory_state(self) -> dict[str, Any]:
        """Return a snapshot of the current memory state."""
        return {
            "turn": self._turn_count,
            "stim": {
                "pages": len(self.stim.pages),
                "capacity": self.stim.capacity,
            },
            "mtem": {
                "sessions": len(self.mtem.sessions),
                "max_sessions": self.mtem.max_sessions,
                "structure_distribution": _count_structures(self.mtem.sessions),
            },
            "ltsm": {
                "entries": len(self.ltsm.entries),
                "max_entries": self.ltsm.max_entries,
                "entry_types": _count_types(self.ltsm.entries),
            },
        }

    def add_user_fact(self, fact: str, entry_type: str = "profile") -> LTSMEntry:
        """Directly inject a known fact about the user into LTSM."""
        return self.ltsm.add_entry(fact, entry_type=entry_type)

    def flush_stim_to_mtem(self) -> None:
        """Force-flush all current STIM pages into MTEM (e.g. at session end)."""
        pages = self.stim.clear()
        if pages:
            self.mtem.ingest_pages(pages)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _count_structures(sessions: list[EpisodicSession]) -> dict[str, int]:
    counts: dict[str, int] = {"linear": 0, "graph": 0, "hierarchical": 0}
    for s in sessions:
        counts[s.structure_type] = counts.get(s.structure_type, 0) + 1
    return counts


def _count_types(entries: list[LTSMEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.entry_type] = counts.get(e.entry_type, 0) + 1
    return counts
