"""IterRet + delta-mem on MemoryAgentBench (default: LongMemEval, "longmemeval_s*").

Each MemoryAgentBench row is ONE long chat history (far longer than a 24 GB GPU
can attend over in one context) with many questions about it. This is the regime
the earlier benchmarks never tested: LoCoMo conversations and Qasper papers fit
in the prompt whole, so "put everything in the prompt" was always available.

Modes (MAB_MODE):
  combined         S = IterRet evidence,   prompt = IterRet evidence
  truncated        S = only what is read,   prompt = most recent MAB_PROMPT_TOKENS of
                                            the history (the long-context baseline)
  truncated_fullS  S = WHOLE history,       prompt = same recent window -- delta-mem's
                                            own claim: S supplies what fell out of the window
  hybrid           S = WHOLE history,       prompt = IterRet evidence

"WHOLE history in S" is written once per history by streaming it through the
model chunk by chunk (osam_workmem.stream_into_state) and restored per question.

Environment:
  MAB_SOURCE          default longmemeval_s*
  MAB_MODE            combined | truncated | truncated_fullS | hybrid
  MAB_PROMPT_TOKENS   recent-window size for the truncated modes (default 32768)
  MAB_MAX_CONTEXTS    first N histories (0 = all)
  MAB_MAX_QUESTIONS   first N questions per history (0 = all)
  MAB_JUDGE           1 = also score with the LLM judge on the graph vLLM (default 1)
  MAB_MAX_EVIDENCE_TOKENS  optional cap on IterRet evidence (default 0 = none)
  MAB_SPLIT           sessions (LongMemEval-aware, dated passages; default) | fixed
  MAB_STATE_DIAG      1 = measure S's memory span after streaming (default 1)
  OSAM_PHASE1_GRANULARITY  message_mean (one write per passage; default) | token
  WORKMEM_OUTPUT_FILE results JSONL (resumable)
  CAIMMS_ADAPTER_DIR  delta-mem adapter
"""
from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.longctx_data import (
    MAB_MAX_NEW_TOKENS, MAB_QUERY_TEMPLATES, doc_key, load_mab, longmemeval_passages, mab_family,
    mab_question_core, mab_scores,
)
from deltamem.workmem.longctx_retrieval import (
    DEFAULT_EVIDENCE_LAYERS, DEFAULT_EVIDENCE_ORDER, evidence_items, get_or_build_doc_graph,
    graph_cache_path, make_backend, present_evidence, retrieve_evidence,
)
from deltamem.workmem.osam_workmem import (
    PHASE1_WRITE_GRANULARITY, answer_from_state, answer_with_modes, stream_into_state,
)
from iterret.doc_memory_builder import chunk_document
from iterret.llm_client import OpenAICompatibleLLMClient

_ROOT = os.environ.get("CAIMMS_ROOT", ".")
_WS = os.environ.get("CAIMMS_WORKSPACE", _ROOT)
MODEL_PATH = os.environ.get("CAIMMS_MODEL_PATH", f"{_WS}/models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = os.environ.get("CAIMMS_ADAPTER_DIR", f"{_WS}/models/delta-mem-adapter")
VLLM_BASE_URL = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

SOURCE = os.environ.get("MAB_SOURCE", "longmemeval_s*")
MODE = os.environ.get("MAB_MODE", "combined")
PROMPT_TOKENS = int(os.environ.get("MAB_PROMPT_TOKENS", "32768"))
MAX_CONTEXTS = int(os.environ.get("MAB_MAX_CONTEXTS", "0"))
MAX_QUESTIONS = int(os.environ.get("MAB_MAX_QUESTIONS", "0"))
JUDGE = os.environ.get("MAB_JUDGE", "1") == "1"
MAX_EVIDENCE_TOKENS = int(os.environ.get("MAB_MAX_EVIDENCE_TOKENS", "0"))
OUTPUT_FILE = os.environ.get("WORKMEM_OUTPUT_FILE", f"{_WS}/outputs/mab_{SOURCE.replace('*', 'star')}_{MODE}.jsonl")
OUT_DIR = Path(OUTPUT_FILE).parent
DATA_DIR = os.environ.get("MAB_DATA_DIR", str(OUT_DIR / "mab_data"))
# How a history is cut into passages (graph nodes, recent window, streamed S):
#   sessions -- LongMemEval-aware: turns grouped within a chat session, every
#               passage prefixed with its session's date (default)
#   fixed    -- generic ~180-word chunks (iterret.doc_memory_builder.chunk_document)
SPLIT = os.environ.get("MAB_SPLIT", "sessions")
GRAPH_CACHE_DIR = Path(os.environ.get("MAB_GRAPH_CACHE_DIR",
                                      str(OUT_DIR / "mab_graph_cache" / f"{SOURCE.replace('*', 'star')}_{SPLIT}")))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("WORKMEM_MAX_CONSECUTIVE_FAILURES", "5"))

# Memory-span check (on by default; a few seconds per history): rebuild S from
# only the last k passages and compare it with the whole-history S. Near-identical
# means S has forgotten everything older -- the write gate keeps ~82% of S per
# write (sigmoid(beta_bias_init=-1.5)), i.e. a span of ~5-20 writes.
STATE_DIAG = os.environ.get("MAB_STATE_DIAG", "1") == "1"

MODES = ("combined", "truncated", "truncated_fullS", "hybrid")
if MODE not in MODES:
    raise SystemExit(f"[FATAL] unknown MAB_MODE={MODE!r} (expected {' | '.join(MODES)})")
FAMILY = mab_family(SOURCE)
if FAMILY not in MAB_QUERY_TEMPLATES:
    raise SystemExit(f"[FATAL] no query template for MAB source {SOURCE!r}")
if SPLIT not in ("sessions", "fixed"):
    raise SystemExit(f"[FATAL] unknown MAB_SPLIT={SPLIT!r} (expected sessions | fixed)")
NEEDS_RETRIEVAL = MODE in ("combined", "hybrid")


def split_history(context: str) -> list:
    if SPLIT == "sessions" and FAMILY == "longmemeval":
        passages = longmemeval_passages(context)
        if passages:
            return passages
        print("[split] WARNING: history is not in LongMemEval session format -- using fixed chunks", flush=True)
    return chunk_document(context)
NEEDS_FULL_STATE = MODE in ("truncated_fullS", "hybrid")


def _load_done(path: str) -> dict:
    done = {}
    if Path(path).exists():
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("prediction") == "" and not row.get("skipped"):
                    continue
                done[(row["ctx_idx"], row["q_idx"])] = row
    return done


def _write(row: dict) -> None:
    with open(OUTPUT_FILE, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _recent_window(passages: list, tokenizer, budget: int) -> list:
    """Most recent passages (document order kept) that fit in ``budget`` tokens."""
    kept, used = [], 0
    for p in reversed(passages):
        n = len(tokenizer(p, add_special_tokens=False)["input_ids"]) + 6
        if kept and used + n > budget:
            break
        kept.append(p)
        used += n
    return list(reversed(kept))


def _state_similarity(a: dict, b: dict) -> tuple:
    """(cosine of all layers' S concatenated, mean relative difference per layer)."""
    keys = sorted(set(a) & set(b))
    va = torch.cat([a[k].float().flatten() for k in keys])
    vb = torch.cat([b[k].float().flatten() for k in keys])
    cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
    rel = [float((a[k].float() - b[k].float()).norm() / (a[k].float().norm() + 1e-12)) for k in keys]
    return cos, sum(rel) / max(1, len(rel))


def _summary(rows: list) -> None:
    if not rows:
        print("No results.", flush=True)
        return
    def line(label, rs):
        ans = [r for r in rs if not r.get("skipped")]
        msg = (f"{label:28} n={len(rs):>4}  substringEM={sum(r['substring_exact_match'] for r in rs) / len(rs):.4f}"
               f"  EM={sum(r['exact_match'] for r in rs) / len(rs):.4f}  F1={sum(r['f1'] for r in rs) / len(rs):.4f}")
        if any("judge_correct" in r for r in rs):
            j = [r.get("judge_correct", False) for r in rs]
            msg += f"  judge={sum(j) / len(j):.4f}"
        msg += f"  skipped={len(rs) - len(ans)}"
        print(msg, flush=True)
    print("=" * 78, flush=True)
    print(f"MemoryAgentBench {SOURCE} | mode={MODE} | prompt_tokens={PROMPT_TOKENS} | adapter={ADAPTER_DIR}", flush=True)
    line("ALL", rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("question_type") or "?"].append(r)
    for t in sorted(by_type):
        line(f"  {t}", by_type[t])
    print("=" * 78, flush=True)


def main() -> None:
    print(f"[init] source={SOURCE} mode={MODE} split={SPLIT} write={PHASE1_WRITE_GRANULARITY} "
          f"prompt_tokens={PROMPT_TOKENS} out={OUTPUT_FILE}", flush=True)
    print(f"[init] adapter={ADAPTER_DIR} graph_cache={GRAPH_CACHE_DIR}", flush=True)
    rows = load_mab(SOURCE, cache_dir=DATA_DIR)
    if MAX_CONTEXTS > 0:
        rows = rows[:MAX_CONTEXTS]
    for r in rows:
        if MAX_QUESTIONS > 0:
            r["questions"] = r["questions"][:MAX_QUESTIONS]
    done = _load_done(OUTPUT_FILE)
    todo_ctx = [r for r in rows if any((r["ctx_idx"], q["q_idx"]) not in done for q in r["questions"])]
    n_q = sum(len(r["questions"]) for r in rows)
    print(f"[init] {len(rows)} histories, {n_q} questions, {len(done)} done, "
          f"{len(todo_ctx)} histories with work left", flush=True)
    for r in rows:  # what the data looks like -- check this on the first run
        types = defaultdict(int)
        for q in r["questions"]:
            types[q.get("question_type") or "?"] += 1
        print(f"[data] history {r['ctx_idx']}: {len(r['context']):,} chars, ~{len(r['context'].split()):,} words, "
              f"{len(r['questions'])} questions {dict(types)}", flush=True)
    if rows:
        print(f"[data] history 0 starts: {rows[0]['context'][:400]!r}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not todo_ctx:
        _summary(list(done.values()))
        return

    llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
    if NEEDS_RETRIEVAL:
        for r in todo_ctx:  # graphs before the model loads (vLLM only)
            if not graph_cache_path(GRAPH_CACHE_DIR, doc_key(r["context"])).exists():
                t0 = time.time()
                print(f"[graphs] building history {r['ctx_idx']} ...", flush=True)
                g, passages, _ = get_or_build_doc_graph(r["context"], doc_key(r["context"]), GRAPH_CACHE_DIR, llm,
                                                        segment_fn=split_history, segmentation=SPLIT)
                print(f"[graphs] history {r['ctx_idx']}: {len(passages)} passages, "
                      f"{g.meta.get('n_semantic', 0)} facts, {time.time() - t0:.0f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
    attach_delta_adapter_in_place(model, Path(ADAPTER_DIR))
    model.eval()
    backend = make_backend() if NEEDS_RETRIEVAL else None
    if JUDGE:
        from iterret.llm_judge import judge_answer

    template = MAB_QUERY_TEMPLATES[FAMILY]
    max_new = MAB_MAX_NEW_TOKENS[FAMILY]
    consecutive_failures = 0
    for r in todo_ctx:
        ctx_idx = r["ctx_idx"]
        passages = split_history(r["context"])  # identical to the graph's passages
        window = _recent_window(passages, tokenizer, PROMPT_TOKENS) if MODE.startswith("truncated") else []
        graph = None
        if NEEDS_RETRIEVAL:
            graph, _, _ = get_or_build_doc_graph(r["context"], doc_key(r["context"]), GRAPH_CACHE_DIR, llm,
                                                 segment_fn=split_history, segmentation=SPLIT)
        snapshot = None
        if NEEDS_FULL_STATE:
            t0 = time.time()
            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            snapshot = stream_into_state(session, passages)
            del session
            torch.cuda.empty_cache()
            print(f"[state] history {ctx_idx}: streamed {len(passages)} passages into S in "
                  f"{time.time() - t0:.0f}s (write granularity {PHASE1_WRITE_GRANULARITY})", flush=True)
            if STATE_DIAG:
                n_win = len(_recent_window(passages, tokenizer, PROMPT_TOKENS))
                for k in sorted({1, 5, 20, n_win}):
                    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
                    tail_state = stream_into_state(session, passages[-k:])
                    del session
                    cos, rel = _state_similarity(snapshot, tail_state)
                    print(f"[state-span] history {ctx_idx}: S(last {k:>4} passages) vs S(all {len(passages)}): "
                          f"cosine {cos:.4f}, mean relative difference {rel:.4f}", flush=True)
                torch.cuda.empty_cache()
        if window:
            print(f"[window] history {ctx_idx}: prompt keeps the last {len(window)}/{len(passages)} passages", flush=True)

        for q in r["questions"]:
            q_idx = q["q_idx"]
            if (ctx_idx, q_idx) in done:
                continue
            question = q["question"]
            formatted_query = template.format(question=question)
            base = {"ctx_idx": ctx_idx, "q_idx": q_idx, "source": SOURCE, "question": question,
                    "answers": q["answers"], "question_type": q.get("question_type"),
                    "mode": MODE, "split": SPLIT, "adapter_dir": ADAPTER_DIR, "prompt_tokens": PROMPT_TOKENS,
                    "write_granularity": PHASE1_WRITE_GRANULARITY,
                    "n_passages": len(passages), "n_window": len(window)}

            evidence, diag = [], {}
            if NEEDS_RETRIEVAL:
                try:
                    # search on the question itself, not its "Current Date: ..." header
                    evidence = retrieve_evidence(mab_question_core(question), graph, backend, llm, diag=diag)
                    items = present_evidence(
                        evidence_items(evidence, diag.get("final_evidence_ids", []), graph),
                        layers=DEFAULT_EVIDENCE_LAYERS, order=DEFAULT_EVIDENCE_ORDER,
                        tokenizer=tokenizer, max_tokens=MAX_EVIDENCE_TOKENS)
                    evidence = [it[0] for it in items]
                    diag["final_evidence_ids"] = [it[1] for it in items]
                except Exception as exc:  # noqa: BLE001 -- not checkpointed, retried next run
                    print(f"[{ctx_idx}.{q_idx}] IterRet FAILED: {exc}", flush=True)
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise SystemExit("[FATAL] consecutive failures -- check vLLM and re-run")
                    continue
                base["n_evidence"] = len(evidence)
                base["retrieval"] = diag
                if not evidence and MODE == "combined":
                    _write({**base, "prediction": "", "skipped": True, "reason": "no_evidence",
                            **mab_scores("", q["answers"]), **({"judge_correct": False} if JUDGE else {})})
                    continue

            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            session.reset()
            try:
                if MODE == "combined":
                    out = answer_with_modes(session, question, s_content=evidence, prompt_content=evidence,
                                            formatted_query=formatted_query, max_new_tokens=max_new)
                elif MODE == "truncated":
                    out = answer_from_state(session, snapshot=None, prompt_content=window,
                                            formatted_query=formatted_query, write_prompt_content=True,
                                            max_new_tokens=max_new)
                elif MODE == "truncated_fullS":
                    out = answer_from_state(session, snapshot=snapshot, prompt_content=window,
                                            formatted_query=formatted_query, write_prompt_content=False,
                                            max_new_tokens=max_new)
                else:  # hybrid
                    out = answer_from_state(session, snapshot=snapshot, prompt_content=evidence,
                                            formatted_query=formatted_query, write_prompt_content=False,
                                            max_new_tokens=max_new)
                prediction = str(out.get("response") or out.get("assistant") or out.get("text") or "").strip() \
                    if isinstance(out, dict) else ""
                if not prediction:
                    prediction = next((m.get("content", "") for m in reversed(session.messages)
                                       if m.get("role") == "assistant"), "").strip()
                osam = (out.get("prompt_output_ratio_stats") or {}) if isinstance(out, dict) else {}
            except Exception as exc:  # noqa: BLE001
                print(f"[{ctx_idx}.{q_idx}] generation FAILED: {exc}", flush=True)
                del session
                torch.cuda.empty_cache()
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit("[FATAL] consecutive generation failures -- aborting (finished rows kept)")
                continue
            consecutive_failures = 0
            del session
            torch.cuda.empty_cache()

            scores = mab_scores(prediction, q["answers"])
            row = {**base, "prediction": prediction, "skipped": False, **scores, "osam_contribution": osam}
            if JUDGE:
                try:
                    row["judge_correct"] = bool(judge_answer(question, q["answers"][0], prediction, llm))
                except Exception as exc:  # noqa: BLE001
                    print(f"[{ctx_idx}.{q_idx}] judge failed: {exc}", flush=True)
                    row["judge_correct"] = False
            _write(row)
            print(f"[{ctx_idx}.{q_idx}] subEM={scores['substring_exact_match']:.0f} F1={scores['f1']:.2f}"
                  f"{' judge=' + str(int(row['judge_correct'])) if JUDGE else ''} pred={prediction[:60]!r} "
                  f"gold={q['answers'][0][:40]!r}", flush=True)
            gc.collect()
        del snapshot

    final = _load_done(OUTPUT_FILE)
    _summary([final[k] for k in sorted(final)
              if k[0] in {r["ctx_idx"] for r in rows}])


if __name__ == "__main__":
    main()
