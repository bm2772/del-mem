
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, TypedDict

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Module = Literal["Planning", "Reflection"]


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def similarity(self, a: List[float], b: List[float]) -> float:
        ...


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """Lazily loads a small sentence-transformers model. Never raises on
    construction -- only ``encode()`` raises, so callers can fall back to
    ``KeywordOverlapEmbeddingBackend`` when this backend can't load.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def encode(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbeddingBackend could not load '{self.model_name}' "
                "(package missing or model not downloaded for offline use)."
            )
        return self._model.encode(text).tolist()

    def similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class KeywordOverlapEmbeddingBackend(EmbeddingBackend):
    """Dependency-free fallback: represents text as a token-count dict and
    compares with cosine similarity over that sparse vector."""

    def encode(self, text: str) -> Dict[str, int]:  # type: ignore[override]
        counts: Dict[str, int] = {}
        for token in text.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if token:
                counts[token] = counts.get(token, 0) + 1
        return counts

    def similarity(self, a: Dict[str, int], b: Dict[str, int]) -> float:  # type: ignore[override]
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
        return dot / (norm_a * norm_b)


def build_default_embedding_backend() -> EmbeddingBackend:
    backend = SentenceTransformerEmbeddingBackend()
    try:
        backend.encode("warmup")
        return backend
    except Exception:
        return KeywordOverlapEmbeddingBackend()


class ExperienceEntry(TypedDict):
    condition: str
    situation: str
    experience: str
    module: Module
    embedding: object  # List[float] or Dict[str, int], depending on backend


class ExperienceBank:
    """Holds D^P and D^R (Eq. 10) plus retrieval (Eq. 12)."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.planning_bank: List[ExperienceEntry] = []
        self.reflection_bank: List[ExperienceEntry] = []

    def _bank_for(self, module: Module) -> List[ExperienceEntry]:
        return self.planning_bank if module == "Planning" else self.reflection_bank

    def add_experience(self, condition: str, situation: str, experience: str, module: Module) -> ExperienceEntry:
        embedding = self.backend.encode(condition + " " + situation)
        entry: ExperienceEntry = {
            "condition": condition,
            "situation": situation,
            "experience": experience,
            "module": module,
            "embedding": embedding,
        }
        self._bank_for(module).append(entry)
        return entry

    def retrieve(self, condition: str, situation: str, module: Module, k: int = 3) -> List[ExperienceEntry]:
        bank = self._bank_for(module)
        if not bank:
            return []
        query_embedding = self.backend.encode(condition + " " + situation)
        scored = [(self.backend.similarity(query_embedding, entry["embedding"]), entry) for entry in bank]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        def strip_embedding(entries: List[ExperienceEntry]) -> List[dict]:
            return [{k: v for k, v in entry.items() if k != "embedding"} for entry in entries]

        return {
            "planning_bank": strip_embedding(self.planning_bank),
            "reflection_bank": strip_embedding(self.reflection_bank),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str, backend: EmbeddingBackend) -> "ExperienceBank":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        bank = cls(backend)
        for entry in data.get("planning_bank", []):
            bank.add_experience(entry["condition"], entry["situation"], entry["experience"], "Planning")
        for entry in data.get("reflection_bank", []):
            bank.add_experience(entry["condition"], entry["situation"], entry["experience"], "Reflection")
        return bank


def empty_experience_bank() -> ExperienceBank:
    """A bank with no entries -- used by offline trajectory collection so
    the bootstrap run is unguided, matching R2-Mem's "collect baseline
    trajectories before any experience exists" setup."""
    return ExperienceBank(KeywordOverlapEmbeddingBackend())


# ----------------------------------------------------------------------
# Condition construction (R2-Mem Eq. 11)
# ----------------------------------------------------------------------
def planning_condition(current_refined_query: str) -> str:
    """c_i = q_i for the Planning bank."""
    return current_refined_query


def reflection_condition(original_query: str, accumulated_evidence: List[str]) -> str:
    """c_i = [Q + m_i] for the Reflection bank."""
    evidence_block = "; ".join(accumulated_evidence)
    return f"{original_query} | evidence so far: {evidence_block}"
