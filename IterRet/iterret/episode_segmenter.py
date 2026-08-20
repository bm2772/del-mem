"""Bridge between the surprise boundary creator and IterRet's episodic layer.

The surprise module (``iterret.surprise``) segments a *token stream* into
events. IterRet's memory, however, is built from a list of dialogue *turns*.
``SurpriseEpisodeSegmenter`` renders the turns into one token stream, runs the
``CAIMMSBoundaryEmitter`` over it to get event boundaries, and snaps those
token-level boundaries back onto whole-turn boundaries -- so an "event" is a
contiguous group of turns. ``memory_builder.build_ctc_graph_from_dialogue``
then makes one episodic node per event instead of per turn.

Snapping to whole turns (rather than slicing mid-turn) keeps speaker
attribution and per-turn timestamps intact and makes exact intra-turn token
offsets irrelevant.

Importing this module is cheap: ``torch``/``transformers``/``iterret.surprise``
are imported lazily inside ``SurpriseEpisodeSegmenter.__init__``, so the rest
of IterRet keeps running without those (heavy) dependencies installed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def build_turn_events(
    cut_token_indices: Sequence[int],
    turn_starts: Sequence[int],
    n_turns: int,
) -> List[List[int]]:
    """Map emitted token cut points to contiguous groups of turn indices.

    ``turn_starts[i]`` is the token index at which turn ``i`` begins in the
    concatenated stream. Each cut (a token index where a new event begins)
    snaps to the turn whose *start* token is nearest, yielding the set of turns
    that begin a new event. Turn 0 always begins the first event; cuts that
    snap onto an already-chosen turn collapse (dedupe). The returned events are
    consecutive, non-empty, and together cover ``range(n_turns)`` -- including
    the trailing tail after the last emitted boundary, which forms the final
    event automatically.
    """
    if n_turns <= 0:
        return []
    starts = {0}
    for cut in cut_token_indices:
        nearest_turn = min(range(n_turns), key=lambda t: abs(turn_starts[t] - cut))
        starts.add(nearest_turn)
    boundaries = sorted(starts)
    events: List[List[int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n_turns
        events.append(list(range(start, end)))
    return events


class SurpriseEpisodeSegmenter:
    """Segment dialogue turns into surprise-bounded events using a local HF model.

    One loaded model can be reused across many conversations: ``segment()``
    resets the emitter's rolling state at the start of every call, so boundary
    history never bleeds between conversations.

    Parameters mirror the surprise module's knobs:
    - ``gamma``: surprise threshold strictness (higher -> larger, fewer events).
    - ``min_block_size``: minimum event size in tokens.
    - ``similarity_refinement``: enable Stage-2 KV-modularity boundary refinement
      (disable if your ``transformers`` version's KV cache isn't indexable in the
      legacy ``past_key_values[layer][0]`` form the surprise module expects).
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct",
        *,
        gamma: float = 1.5,
        min_block_size: int = 8,
        n_local: int = 4096,
        n_init: int = 128,
        similarity_refinement: bool = True,
        chunk_size: int = 2048,
        device=None,
    ) -> None:
        import torch  # noqa: PLC0415 -- deliberately lazy: keep torch optional
        from transformers import AutoTokenizer  # noqa: PLC0415

        from .surprise import CAIMMSBoundaryEmitter, StatefulSurpriseBoundary

        self._torch = torch
        self._StatefulSurpriseBoundary = StatefulSurpriseBoundary

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.chunk_size = chunk_size
        self.min_block_size = min_block_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.emitter = CAIMMSBoundaryEmitter(
            model_name_or_path=model_name_or_path,
            surprisal_threshold_gamma=gamma,
            n_local=n_local,
            n_init=n_init,
            similarity_refinement=similarity_refinement,
            device=device,
        )
        self.emitter.eval()

        # CAIMMSBoundaryEmitter doesn't expose min_block_size; it lives on the
        # tracker. Keep the kwargs so reset() can rebuild an identical tracker.
        self._tracker_kwargs = dict(
            batch_size=1,
            surprisal_threshold_gamma=gamma,
            n_local=n_local,
            n_init=n_init,
            similarity_refinement=similarity_refinement,
            similarity_metric="modularity",
            min_block_size=min_block_size,
            device=device,
            dtype=torch.float32,
        )
        self.reset()

    def reset(self) -> None:
        """Clear all rolling boundary state (called at the start of segment())."""
        self.emitter.boundary_tracker = self._StatefulSurpriseBoundary(**self._tracker_kwargs)
        self.emitter.last_boundary_idx = 0
        self.emitter.new_episodes = []
        # No carried predictor at the start of a fresh stream -> the first token
        # is treated as unconditioned (see CAIMMSBoundaryEmitter.forward).
        self.emitter._prev_last_logit = None

    def _render_and_tokenize(self, turns: Sequence[dict]) -> Tuple[List[int], List[int]]:
        """Return (concatenated token ids, per-turn start offsets)."""
        all_ids: List[int] = []
        turn_starts: List[int] = []
        for turn in turns:
            turn_starts.append(len(all_ids))
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            rendered = f"{speaker}: {text}\n"
            ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if not ids:  # never let a turn occupy zero tokens
                fallback = self.tokenizer.eos_token_id
                ids = [fallback if fallback is not None else 0]
            all_ids.extend(ids)
        return all_ids, turn_starts

    def emit_episodes(self, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
        """Run the surprise emitter over a raw token-id stream and return the
        (start, end) token spans it emits. Resets rolling state first, so calls
        are independent. Shared by dialogue segmentation (which snaps the spans'
        ends to turn boundaries) and document segmentation (which uses the raw
        spans directly, EM-LLM's native mode)."""
        self.reset()
        torch = self._torch
        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        total = input_ids.shape[1]

        past_key_values = None
        with torch.no_grad():
            for start in range(0, total, self.chunk_size):
                chunk = input_ids[:, start:start + self.chunk_size]
                outputs = self.emitter(
                    input_ids=chunk,
                    labels=chunk,  # teacher-force surprisal against the real text
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        return self.emitter.get_new_episodes()

    def segment(self, turns: Sequence[dict]) -> List[List[int]]:
        """Segment ``turns`` into events; each event is a list of turn indices."""
        if not turns:
            return []
        n_turns = len(turns)
        all_ids, turn_starts = self._render_and_tokenize(turns)
        if not all_ids:
            return [list(range(n_turns))]

        episodes = self.emit_episodes(all_ids)
        cut_indices = sorted({end for _start, end in episodes})
        return build_turn_events(cut_indices, turn_starts, n_turns)


# ----------------------------------------------------------------------
# Shared CLI wiring, reused by build_memory.py / run_locomo_eval.py /
# pipeline_walkthrough.py so the surprise flags stay identical across them.
# ----------------------------------------------------------------------
def add_surprise_cli_args(parser) -> None:
    group = parser.add_argument_group("surprise-based episodic segmentation")
    group.add_argument(
        "--surprise-segmentation", action="store_true",
        help="Segment dialogue into surprise-bounded events (one episodic node per event) "
             "instead of one per turn. Requires torch+transformers and a local HF model; "
             "a GPU is strongly recommended.")
    group.add_argument(
        "--surprise-model", default="Qwen/Qwen3-4B-Instruct",
        help="HF model id used to compute token surprisal (default %(default)s).")
    group.add_argument(
        "--surprise-gamma", type=float, default=1.5,
        help="Surprise-threshold strictness; higher -> fewer, larger events (default %(default)s).")
    group.add_argument(
        "--surprise-min-block-size", type=int, default=8,
        help="Minimum event size in tokens (default %(default)s).")
    group.add_argument(
        "--surprise-no-refinement", action="store_true",
        help="Disable Stage-2 KV graph-modularity boundary refinement (Stage-1 surprise only).")
    group.add_argument(
        "--surprise-device", default=None,
        help="Device for the surprise model: cuda/mps/cpu (default: auto-detect, "
             "preferring cuda, then mps, then cpu).")


def segmenter_from_args(args):
    """Construct a ``SurpriseEpisodeSegmenter`` iff ``--surprise-segmentation``
    was passed (which is also when torch/transformers get imported); else None."""
    if not getattr(args, "surprise_segmentation", False):
        return None
    return SurpriseEpisodeSegmenter(
        args.surprise_model,
        gamma=args.surprise_gamma,
        min_block_size=args.surprise_min_block_size,
        similarity_refinement=not getattr(args, "surprise_no_refinement", False),
        device=getattr(args, "surprise_device", None),
    )
