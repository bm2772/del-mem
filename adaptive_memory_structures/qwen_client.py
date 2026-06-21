"""
qwen_client.py – Centralised Qwen3-4B Interface
=================================================
Single place to load, configure, and call the Qwen3-4B model.
Every other module (pipeline.py, rewards.py, …) imports from here.

Features
--------
- Singleton model registry  – load once, share everywhere, never double-load
- QwenClient                – unified generate / embed / chat API
- Batched generation        – process multiple prompts in one forward pass
- Embedding with caching    – LRU cache avoids re-encoding identical strings
- INT8 / INT4 quantisation  – BitsAndBytes support for low-VRAM setups
- Thinking mode toggle      – Qwen3's extended chain-of-thought on/off per call
- Dry-run / mock mode       – full API surface with no model required (tests)
- Device auto-detection     – CUDA → MPS → CPU

Usage
-----
    # Load once at startup
    from qwen_client import QwenClient
    qwen = QwenClient.load("Qwen/Qwen3-4B")

    # Generate a response
    reply = qwen.generate("What is the capital of France?")

    # Embed a sentence
    vec = qwen.embed("Paris is a city in France.")

    # Chat with a system prompt
    reply = qwen.chat(
        user="Summarise this text.",
        system="You are a concise summariser.",
    )

    # Batch generation (returns list[str])
    replies = qwen.generate_batch(["Q1?", "Q2?", "Q3?"])

    # Reuse across modules – same object, no second model load
    from qwen_client import get_client
    qwen = get_client()   # returns the already-loaded singleton
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton registry
# ─────────────────────────────────────────────────────────────────────────────

_registry: dict[str, "QwenClient"] = {}
_registry_lock = Lock()


def get_client(key: str = "default") -> "QwenClient":
    """
    Return the already-registered QwenClient for ``key``.
    Raises RuntimeError if no client has been loaded under that key.
    """
    with _registry_lock:
        if key not in _registry:
            raise RuntimeError(
                f"No QwenClient loaded under key '{key}'. "
                "Call QwenClient.load(...) first."
            )
        return _registry[key]


def is_loaded(key: str = "default") -> bool:
    """Return True if a client is already registered under ``key``."""
    with _registry_lock:
        return key in _registry


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QwenConfig:
    """
    All knobs for loading and running Qwen3-4B.

    Parameters
    ----------
    model_path
        HuggingFace model ID or local directory path.
    device
        "auto" lets HuggingFace pick (CUDA if available).
        Override with "cuda", "cuda:1", "mps", or "cpu".
    dtype
        "auto" uses bfloat16 on CUDA, float32 on CPU.
        Pass "float16" or "bfloat16" to force a specific dtype.
    quantisation
        None        – no quantisation (default)
        "int8"      – LLM.int8() via BitsAndBytes (≈ 50% VRAM)
        "int4"      – QLoRA 4-bit via BitsAndBytes (≈ 25% VRAM)
    max_new_tokens
        Default token budget for generation calls.
    temperature
        Default sampling temperature.  0 = greedy.
    top_p
        Nucleus sampling probability (only used when temperature > 0).
    embed_max_length
        Tokeniser truncation for embedding calls.
    embed_cache_size
        Number of (text → embedding) pairs to cache in memory.
        Set to 0 to disable caching.
    enable_thinking
        Qwen3's extended chain-of-thought mode.  Disable for speed.
    trust_remote_code
        Passed to from_pretrained.
    """
    model_path: str         = "Qwen/Qwen3-4B"
    device: str             = "auto"
    dtype: str              = "auto"
    quantisation: str | None = None        # None | "int8" | "int4"
    max_new_tokens: int     = 512
    temperature: float      = 0.0
    top_p: float            = 0.9
    embed_max_length: int   = 512
    embed_cache_size: int   = 1024
    enable_thinking: bool   = False        # False = fast non-thinking mode
    trust_remote_code: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Generation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Structured output of a single generate() call."""
    text: str               = ""
    prompt_tokens: int      = 0
    completion_tokens: int  = 0
    latency_ms: float       = 0.0
    thinking: str           = ""   # only populated when enable_thinking=True


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve "auto" to the best available device string."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# QwenClient
# ─────────────────────────────────────────────────────────────────────────────

class QwenClient:
    """
    Unified interface for Qwen3-4B generation and embedding.

    Do not instantiate directly in production – use ``QwenClient.load()``
    which handles singleton registration so the model is loaded only once.
    """

    # ------------------------------------------------------------------
    # Construction / singleton
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: QwenConfig,
        device: str,
        dry_run: bool = False,
    ):
        self._model     = model
        self._tokenizer = tokenizer
        self._config    = config
        self._device    = device
        self._dry_run   = dry_run

        # embedding cache: sha256(text) → np.ndarray
        self._embed_cache: dict[str, np.ndarray] = {}
        self._cache_hits   = 0
        self._cache_misses = 0

        # call counters
        self._gen_calls   = 0
        self._embed_calls = 0

    @classmethod
    def load(
        cls,
        model_path: str = "Qwen/Qwen3-4B",
        config: QwenConfig | None = None,
        registry_key: str = "default",
        dry_run: bool = False,
        force_reload: bool = False,
    ) -> "QwenClient":
        """
        Load Qwen3-4B and register it as a singleton.

        Parameters
        ----------
        model_path
            HuggingFace ID or local path. Overrides config.model_path
            when config is also provided.
        config
            Full QwenConfig object.  Uses defaults if None.
        registry_key
            Name under which to store this client.  Use different keys
            to maintain multiple models simultaneously.
        dry_run
            Skip model loading; all API calls return mock outputs.
        force_reload
            Re-load even if a client is already registered under the key.

        Returns
        -------
        QwenClient registered (and cached) under ``registry_key``.
        """
        with _registry_lock:
            if not force_reload and registry_key in _registry:
                logger.info(
                    f"[QwenClient] Reusing existing client '{registry_key}' "
                    f"(model={_registry[registry_key]._config.model_path})"
                )
                return _registry[registry_key]

            cfg = config or QwenConfig(model_path=model_path)
            if model_path != "Qwen/Qwen3-4B":        # explicit override
                cfg.model_path = model_path

            device = _resolve_device(cfg.device)

            if dry_run:
                client = cls(None, None, cfg, device, dry_run=True)
                _registry[registry_key] = client
                logger.info(f"[QwenClient] Dry-run client registered as '{registry_key}'.")
                return client

            model, tokenizer = cls._load_model(cfg, device)
            client = cls(model, tokenizer, cfg, device, dry_run=False)
            _registry[registry_key] = client
            logger.info(
                f"[QwenClient] '{registry_key}' loaded "
                f"({cfg.model_path} on {device})."
            )
            return client

    @staticmethod
    def _load_model(cfg: QwenConfig, device: str) -> tuple[Any, Any]:
        """Internal: load tokenizer + model with optional quantisation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required: "
                "pip install transformers torch"
            ) from exc

        logger.info(f"[QwenClient] Loading tokenizer from {cfg.model_path} …")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=cfg.trust_remote_code
        )

        # dtype
        if cfg.dtype == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = getattr(torch, cfg.dtype, torch.float32)

        # quantisation config
        bnb_config = None
        if cfg.quantisation in ("int8", "int4"):
            try:
                from transformers import BitsAndBytesConfig
                if cfg.quantisation == "int8":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed – loading without quantisation. "
                    "Install with: pip install bitsandbytes"
                )

        logger.info(
            f"[QwenClient] Loading model "
            f"(quantisation={cfg.quantisation or 'none'}, dtype={cfg.dtype}) …"
        )
        load_kwargs: dict[str, Any] = dict(
            trust_remote_code=cfg.trust_remote_code,
            device_map=device,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **load_kwargs)
        model.eval()
        logger.info("[QwenClient] Model ready.")
        return model, tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input_ids(self, messages: list[dict[str, str]]) -> Any:
        """Tokenise a message list using Qwen3's chat template."""
        import torch
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._config.enable_thinking,
        )
        return self._tokenizer(text, return_tensors="pt").to(self._device)

    def _gen_kwargs(
        self,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> dict[str, Any]:
        """Build HuggingFace generation kwargs."""
        t   = temperature if temperature is not None else self._config.temperature
        tok = max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
        p   = top_p if top_p is not None else self._config.top_p

        kwargs: dict[str, Any] = dict(
            max_new_tokens=tok,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if t > 0:
            kwargs.update(do_sample=True, temperature=t, top_p=p)
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _extract_thinking(self, text: str) -> tuple[str, str]:
        """
        Split Qwen3 thinking output from the final answer.
        Returns (thinking_block, answer_text).
        """
        import re
        thinking_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        answer   = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thinking, answer

    # ------------------------------------------------------------------
    # Core: single generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool | None = None,
    ) -> GenerationResult:
        """
        Generate a response to ``prompt``.

        Parameters
        ----------
        prompt          : user message text
        system          : optional system prompt
        max_new_tokens  : overrides config default
        temperature     : overrides config default (0 = greedy)
        top_p           : overrides config default
        enable_thinking : overrides config default for this call only

        Returns
        -------
        GenerationResult with .text and timing info.
        """
        self._gen_calls += 1

        if self._dry_run or self._model is None:
            return GenerationResult(
                text=f"[DRY-RUN] prompt={prompt[:80]}…",
                latency_ms=0.0,
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # temporarily override thinking if requested
        orig_thinking = self._config.enable_thinking
        if enable_thinking is not None:
            self._config.enable_thinking = enable_thinking

        try:
            import torch
            inputs = self._build_input_ids(messages)
            n_prompt = inputs["input_ids"].shape[-1]

            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    **self._gen_kwargs(max_new_tokens, temperature, top_p),
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            new_ids = output_ids[0][n_prompt:]
            raw     = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            thinking, answer = self._extract_thinking(raw)
            return GenerationResult(
                text=answer,
                prompt_tokens=n_prompt,
                completion_tokens=len(new_ids),
                latency_ms=latency_ms,
                thinking=thinking,
            )
        finally:
            self._config.enable_thinking = orig_thinking

    # ------------------------------------------------------------------
    # Convenience: chat with system prompt
    # ------------------------------------------------------------------

    def chat(
        self,
        user: str,
        system: str | None = None,
        **generate_kwargs: Any,
    ) -> str:
        """
        Convenience wrapper – returns just the answer string.

        Example
        -------
            reply = qwen.chat("Summarise this.", system="Be concise.")
        """
        return self.generate(user, system=system, **generate_kwargs).text

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """
        Generate responses for multiple prompts.

        Prompts are processed in a single batched forward pass when the
        tokeniser supports padding; falls back to sequential calls if not.

        Parameters
        ----------
        prompts  : list of user messages
        system   : single system prompt applied to all messages (optional)
        Returns
        -------
        list[GenerationResult] in the same order as ``prompts``.
        """
        if not prompts:
            return []

        if self._dry_run or self._model is None:
            return [
                GenerationResult(text=f"[DRY-RUN] prompt={p[:60]}…")
                for p in prompts
            ]

        try:
            import torch
        except ImportError:
            return [self.generate(p, system=system) for p in prompts]

        # build chat-formatted text for each prompt
        formatted: list[str] = []
        for p in prompts:
            msgs: list[dict[str, str]] = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            formatted.append(
                self._tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self._config.enable_thinking,
                )
            )

        # tokenise with left-padding so generated tokens align on the right
        orig_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            n_prompt = inputs["input_ids"].shape[1]
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    **self._gen_kwargs(max_new_tokens, temperature, top_p),
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            results: list[GenerationResult] = []
            for i, out in enumerate(output_ids):
                new_ids = out[n_prompt:]
                raw     = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                thinking, answer = self._extract_thinking(raw)
                results.append(GenerationResult(
                    text=answer,
                    prompt_tokens=n_prompt,
                    completion_tokens=len(new_ids),
                    latency_ms=latency_ms / len(prompts),   # amortised
                    thinking=thinking,
                ))
            return results

        except Exception as exc:
            logger.warning(
                f"[QwenClient] Batch generation failed ({exc}), "
                "falling back to sequential."
            )
            return [self.generate(p, system=system) for p in prompts]
        finally:
            self._tokenizer.padding_side = orig_padding_side

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed ``text`` using Qwen3-4B mean-pooling over the last hidden state.

        Results are cached by SHA-256 of the input text (up to
        ``config.embed_cache_size`` entries).

        Parameters
        ----------
        text      : input string to embed
        use_cache : set False to bypass the cache for this call

        Returns
        -------
        Unit-norm float32 numpy array of shape (hidden_dim,).
        """
        self._embed_calls += 1

        if self._dry_run or self._model is None:
            from memory_layers import DummyEmbedder
            return DummyEmbedder().encode(text)

        # cache lookup
        cache_key = hashlib.sha256(text.encode()).hexdigest() if use_cache else None
        if cache_key and cache_key in self._embed_cache:
            self._cache_hits += 1
            return self._embed_cache[cache_key]

        self._cache_misses += 1

        try:
            import torch
        except ImportError:
            from memory_layers import DummyEmbedder
            return DummyEmbedder().encode(text)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.embed_max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]                  # (1, seq, dim)
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            emb    = (hidden * mask).sum(1) / mask.sum(1)       # (1, dim)
            emb    = emb.squeeze(0).float().cpu().numpy()

        norm = np.linalg.norm(emb)
        emb  = emb / (norm + 1e-9)

        # cache store with LRU eviction
        if cache_key and self._config.embed_cache_size > 0:
            if len(self._embed_cache) >= self._config.embed_cache_size:
                # evict oldest (first inserted) key
                oldest = next(iter(self._embed_cache))
                del self._embed_cache[oldest]
            self._embed_cache[cache_key] = emb

        return emb

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed multiple texts.  Processes in a single batched forward pass
        when possible; falls back to sequential if batching fails.

        Returns
        -------
        list of unit-norm float32 arrays, one per input text.
        """
        if not texts:
            return []

        if self._dry_run or self._model is None:
            return [self.embed(t) for t in texts]

        # check cache for all entries first
        results: list[np.ndarray | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            ck = hashlib.sha256(text.encode()).hexdigest()
            if ck in self._embed_cache:
                results.append(self._embed_cache[ck])
                self._cache_hits += 1
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # type: ignore[return-value]

        try:
            import torch
            inputs = self._tokenizer(
                uncached_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.embed_max_length,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
                hidden  = outputs.hidden_states[-1]             # (B, seq, dim)
                mask    = inputs["attention_mask"].unsqueeze(-1).float()
                embs    = (hidden * mask).sum(1) / mask.sum(1)  # (B, dim)
                embs    = embs.float().cpu().numpy()

            for j, idx in enumerate(uncached_indices):
                emb  = embs[j]
                norm = np.linalg.norm(emb)
                emb  = emb / (norm + 1e-9)
                results[idx] = emb
                self._cache_misses += 1

                ck = hashlib.sha256(texts[idx].encode()).hexdigest()
                if self._config.embed_cache_size > 0:
                    if len(self._embed_cache) >= self._config.embed_cache_size:
                        oldest = next(iter(self._embed_cache))
                        del self._embed_cache[oldest]
                    self._embed_cache[ck] = emb

        except Exception as exc:
            logger.warning(
                f"[QwenClient] Batch embedding failed ({exc}), "
                "falling back to sequential."
            )
            for j, idx in enumerate(uncached_indices):
                results[idx] = self.embed(uncached_texts[j])

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Adaptor objects (drop-in replacements for the old classes)
    # ------------------------------------------------------------------

    def as_embedder(self) -> "ClientEmbedderAdaptor":
        """
        Return an object with an ``.encode(text) -> np.ndarray`` method
        compatible with the Embedder protocol used by MTEM / LTSM / rewards.
        """
        return ClientEmbedderAdaptor(self)

    def as_generator(self) -> "ClientGeneratorAdaptor":
        """
        Return an object with a ``.generate(prompt) -> str`` method
        compatible with the old QwenGenerator interface.
        """
        return ClientGeneratorAdaptor(self)

    # ------------------------------------------------------------------
    # Context manager (optional: auto-unload on exit)
    # ------------------------------------------------------------------

    def __enter__(self) -> "QwenClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()

    def unload(self, registry_key: str | None = None) -> None:
        """
        Free model weights from memory.
        If ``registry_key`` is given, also remove from the singleton registry.
        """
        try:
            import torch
            if self._model is not None:
                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("[QwenClient] Model unloaded.")
        except Exception as exc:
            logger.warning(f"[QwenClient] Unload warning: {exc}")

        if registry_key is not None:
            with _registry_lock:
                _registry.pop(registry_key, None)

    # ------------------------------------------------------------------
    # Stats / info
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Runtime statistics."""
        total_embed = self._cache_hits + self._cache_misses
        hit_rate    = self._cache_hits / total_embed if total_embed else 0.0
        return {
            "model_path":      self._config.model_path,
            "device":          self._device,
            "quantisation":    self._config.quantisation,
            "generation_calls": self._gen_calls,
            "embed_calls":     self._embed_calls,
            "cache_hits":      self._cache_hits,
            "cache_misses":    self._cache_misses,
            "cache_hit_rate":  round(hit_rate, 4),
            "cache_size":      len(self._embed_cache),
            "dry_run":         self._dry_run,
        }

    def clear_embed_cache(self) -> int:
        """Evict all cached embeddings. Returns number of entries cleared."""
        n = len(self._embed_cache)
        self._embed_cache.clear()
        return n

    def __repr__(self) -> str:
        return (
            f"QwenClient(model={self._config.model_path!r}, "
            f"device={self._device!r}, "
            f"quantisation={self._config.quantisation!r}, "
            f"dry_run={self._dry_run})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adaptor shims (drop-in for old QwenEmbedder / QwenGenerator)
# ─────────────────────────────────────────────────────────────────────────────

class ClientEmbedderAdaptor:
    """
    Wraps QwenClient to satisfy the ``Embedder`` protocol
    (``encode(text) -> np.ndarray``).

    Use this anywhere an embedder is needed so the underlying model
    is always the shared singleton.
    """

    def __init__(self, client: QwenClient) -> None:
        self._client = client

    def encode(self, text: str) -> np.ndarray:
        return self._client.embed(text)


class ClientGeneratorAdaptor:
    """
    Wraps QwenClient to provide the old ``generate(prompt) -> str`` interface,
    so existing call sites in pipeline.py / rewards.py continue to work
    without changes.
    """

    def __init__(
        self,
        client: QwenClient,
        system: str | None = None,
    ) -> None:
        self._client = client
        self._system = system

    def generate(self, prompt: str) -> str:
        return self._client.chat(prompt, system=self._system)
