"""A/B: does Phase 1 at message_mean (SSW) hurt, given a TSW-trained adapter?

The released adapter (models/delta-mem-adapter) was trained with
memory_write_granularity="token" (model card: "Variant: TSW, Write granularity:
token"). The two-phase OSAM design writes Phase 1 at "message_mean", which is
the paper's SSW strategy -- a write path those weights never saw in training.
This script measures how much that costs, so the decision to ship two-phase (or
to train an SSW adapter first) rests on a number instead of an argument.

Design: retrieval runs ONCE per question and BOTH arms generate from the exact
same evidence list, in the same order, with the same prompt and greedy decoding.
Phase 2 is "token" in both arms (answer_with_osam sets it). So the only variable
in the whole pipeline is the Phase 1 write granularity, and the comparison is
paired -- every question contributes a matched pair, which is far more sensitive
than comparing two independent runs and costs roughly half as much.

Usage (on the cluster, with the vLLM server already up -- same as the main eval):

    CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.ab_write_granularity

    AB_MAX_SAMPLES=1                      # conversations to cover (default 1)
    AB_OUTPUT_FILE=outputs/ab_gran.jsonl  # resumable checkpoint
    AB_ARMS=token,message_mean            # granularities to compare
    AB_MAX_NEW_TOKENS=                    # unset = production behaviour
"""
from __future__ import annotations

import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.eval.locomo_protocol import (
    ADVERSARIAL_CATEGORY,
    CATEGORY_DISPLAY_NAMES,
    score_locomo_prediction,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.evidence_filter import filter_evidence_by_relevance
from deltamem.workmem.iterret_bridge import get_iterret_evidence
from deltamem.workmem.osam_workmem import (
    answer_with_osam,
    maybe_narrow_evidence,
    populate_osam_from_evidence,
)
from deltamem.workmem.eval_locomo_iterret_mock import (
    ADAPTER_DIR,
    DATA_FILE,
    MODEL_PATH,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    extract_prediction,
    extract_session_nums,
    gold_answer_of,
)
from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue

MAX_SAMPLES = int(os.environ.get("AB_MAX_SAMPLES", "1"))
OUTPUT_FILE = os.environ.get("AB_OUTPUT_FILE", "outputs/ab_write_granularity.jsonl")
ARMS = [a.strip() for a in os.environ.get("AB_ARMS", "token,message_mean").split(",") if a.strip()]
_max_new = os.environ.get("AB_MAX_NEW_TOKENS")
GEN_KWARGS = {"max_new_tokens": int(_max_new)} if _max_new else {}


def build_turns(conv_block: dict) -> list:
    turns = []
    for session_num in extract_session_nums(conv_block):
        session_ts = conv_block.get(f"session_{session_num}_date_time", f"Session {session_num}")
        for dialog in conv_block.get(f"session_{session_num}", []):
            speaker = dialog.get("speaker", "Unknown")
            text = f"[{session_ts}] {speaker}: {dialog.get('text', '')}"
            turns.append(DialogueTurn(speaker=speaker, text=text, time=str(session_ts)))
    return turns


def load_done(path: str) -> tuple[set, list]:
    done, rows = set(), []
    p = Path(path)
    if not p.exists():
        return done, rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            done.add((row["sample_idx"], row["q_idx"]))
            rows.append(row)
        except (json.JSONDecodeError, KeyError):
            pass
    return done, rows


def paired_bootstrap(diffs: list, n_boot: int = 10000, seed: int = 0) -> tuple:
    """95% CI on the mean paired difference. Non-parametric, no scipy needed."""
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (
        sum(diffs) / n,
        means[int(0.025 * n_boot)],
        means[int(0.975 * n_boot)],
    )


def main() -> None:
    print(f"[init] arms={ARMS}  max_samples={MAX_SAMPLES}  out={OUTPUT_FILE}", flush=True)
    if len(ARMS) < 2:
        raise SystemExit("AB_ARMS needs at least two granularities to compare.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
    )
    attach_delta_adapter_in_place(model, Path(ADAPTER_DIR))
    model.eval()
    print("[init] model + adapter ready", flush=True)

    samples = json.loads(Path(DATA_FILE).read_text())
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    # Share the main eval's graph cache: the graphs are identical and rebuilding
    # one costs ~600 vLLM calls.
    cache_dir = Path(OUTPUT_FILE).parent / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    done, rows = load_done(OUTPUT_FILE)
    if done:
        print(f"[resume] {len(done)} questions already measured", flush=True)

    graph_llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
    question_llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)

    for sample_idx, sample in enumerate(samples[:MAX_SAMPLES]):
        questions = sample.get("qa", [])
        turns = build_turns(sample.get("conversation", {}))
        if not questions or not turns:
            continue

        cache_path = cache_dir / f"sample_{sample_idx}.json"
        if cache_path.exists():
            graph = CueTagContentGraph.load(str(cache_path))
        else:
            print(f"[sample {sample_idx}] building graph ({len(turns)} turns)...", flush=True)
            graph = build_ctc_graph_from_dialogue(turns, graph_llm)
            graph.save(str(cache_path))
        bank = ExperienceBank(build_default_embedding_backend())
        graph.attach_embedder(bank.backend)
        print(f"[sample {sample_idx}] graph ready: {len(graph.contents)} nodes", flush=True)

        for q_idx, question in enumerate(questions):
            if (sample_idx, q_idx) in done:
                continue
            try:
                if int(question.get("category", 0)) == ADVERSARIAL_CATEGORY:
                    continue
            except (TypeError, ValueError):
                pass

            q_text = question.get("question", "")

            # --- retrieval: ONCE, shared by every arm ---
            try:
                evidence = get_iterret_evidence(q_text, graph, bank, question_llm, max_iterations=5)
            except Exception as exc:
                print(f"[{sample_idx}.{q_idx}] retrieval failed: {exc}", flush=True)
                continue
            if evidence:
                evidence = filter_evidence_by_relevance(
                    q_text, evidence, bank.backend.encode, threshold=0.30
                )
                evidence = maybe_narrow_evidence(q_text, evidence, bank.backend.encode)
            if not evidence:
                # Both arms would score 0 identically -- no signal, skip.
                continue

            # --- generation: once per arm, from identical evidence ---
            row = {
                "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                "gold_answer": gold_answer_of(question), "category": question.get("category"),
                "n_evidence": len(evidence), "arms": {},
            }
            for gran in ARMS:
                session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
                session.reset()
                stats = populate_osam_from_evidence(session, evidence, write_granularity=gran)
                try:
                    out = answer_with_osam(session, q_text, **GEN_KWARGS)
                    prediction = extract_prediction(out, session)
                except Exception as exc:
                    print(f"[{sample_idx}.{q_idx}:{gran}] generation failed: {exc}", flush=True)
                    prediction = ""
                row["arms"][gran] = {
                    "prediction": prediction,
                    "score": score_locomo_prediction(question, prediction),
                    # Proves the two arms produced genuinely different memory states
                    # rather than silently taking the same code path.
                    "state_stats": {k: float(v) for k, v in (stats or {}).items()},
                }
                del session
                torch.cuda.empty_cache()

            with open(OUTPUT_FILE, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            rows.append(row)
            done.add((sample_idx, q_idx))
            summary = "  ".join(f"{g}={row['arms'][g]['score']:.3f}" for g in ARMS)
            print(f"[{sample_idx}.{q_idx}] n_ev={len(evidence)}  {summary}", flush=True)

        del graph, bank
        torch.cuda.empty_cache()

    report(rows)


def report(rows: list) -> None:
    rows = [r for r in rows if r.get("arms") and all(a in r["arms"] for a in ARMS)]
    if not rows:
        print("\nNo paired results.", flush=True)
        return

    base, test = ARMS[0], ARMS[1]
    print(f"\n{'=' * 68}")
    print(f"PAIRED A/B  --  Phase 1 write granularity   (n = {len(rows)} questions)")
    print(f"  baseline = {base!r}   (matches the TSW adapter's training)")
    print(f"  test     = {test!r}")
    print("=" * 68)

    for arm in ARMS:
        scores = [r["arms"][arm]["score"] for r in rows]
        print(f"  {arm:>14s}  F1 = {statistics.mean(scores):.4f}")

    diffs = [r["arms"][test]["score"] - r["arms"][base]["score"] for r in rows]
    mean_d, lo, hi = paired_bootstrap(diffs)
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    identical = sum(
        1 for r in rows
        if r["arms"][base]["prediction"] == r["arms"][test]["prediction"]
    )

    print(f"\n  mean paired diff ({test} - {base}) = {mean_d:+.4f}")
    print(f"  95% bootstrap CI                  = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {test} better on {wins}, worse on {losses}, tied on {len(diffs) - wins - losses}")
    print(f"  identical predictions: {identical}/{len(rows)} ({identical / len(rows) * 100:.1f}%)")

    if identical == len(rows):
        print("\n  !! Every prediction is identical -- the two arms took the SAME code")
        print("     path. Phase 1 is falling back to token writes in both. Check that")
        print("     evidence is ingested with role='user' (see populate_osam_from_evidence).")

    print("\n  per category:")
    by = defaultdict(list)
    for r in rows:
        try:
            by[int(r["category"])].append(r)
        except (TypeError, ValueError):
            continue
    for cat in sorted(by):
        rs = by[cat]
        b = statistics.mean(r["arms"][base]["score"] for r in rs)
        t = statistics.mean(r["arms"][test]["score"] for r in rs)
        print(f"    [{CATEGORY_DISPLAY_NAMES.get(cat, cat):11s}] n={len(rs):4d}  "
              f"{base}={b:.4f}  {test}={t:.4f}  diff={t - b:+.4f}")

    print(f"\n{'-' * 68}")
    if lo <= 0.0 <= hi:
        print("  VERDICT: no significant difference. Running two-phase at "
              f"{test!r} costs nothing measurable on this sample -- ship it and")
        print("           note the train/inference caveat.")
    elif hi < 0.0:
        print(f"  VERDICT: {test!r} is significantly WORSE. The TSW adapter does not")
        print("           transfer to segment writes. Two-phase needs an SSW adapter.")
    else:
        print(f"  VERDICT: {test!r} is significantly BETTER, despite the mismatch.")
    print(f"{'-' * 68}\n", flush=True)


if __name__ == "__main__":
    main()
