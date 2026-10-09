"""Build delta-mem SFT episodes from IterRet-retrieved evidence (Qasper train).

Why: the released adapter is trained on CONTINUOUS documents, with S written
from the older part of the context while the model attends only the recent
part -- S and the prompt never hold the same text. The `combined` pipeline does
the opposite: S and the prompt both hold the same short, reordered IterRet
evidence set. This builds training episodes in that regime, with the evidence
produced by exactly the eval's code path (longctx_retrieval), so retraining
targets what delta-mem is actually fed at inference.

Each output line (delta_sft_experimental --train-file):
  {"messages": [{"role": "user", "content": <evidence 1>}, ...,
                {"role": "user", "content": <LongBench qasper query block>},
                {"role": "assistant", "content": <gold answer>}],
   "paper_id", "question_id", "n_evidence", "answer_in_evidence"}
Train with --episode-recent-messages 1 so the evidence messages are the write
history and the query is the read turn (see scripts/train_iterret_osam.sh).

Second step, `prepare`, turns those rows into the trainer's input file with the
SAME evidence presentation the eval uses (longctx_retrieval.present_evidence;
default: passages only, capped to the token budget from the most relevant end,
then put back in document order). The trainer truncates write history from the
front, so the cap has to happen here:
  python -m deltamem.workmem.build_iterret_sft_data prepare \
      --in sft_iterret_qasper.jsonl --out sft_train.jsonl \
      --tokenizer <model dir> --max-write-tokens 1536

Environment:
  SFT_QASPER_TRAIN   raw qasper-train-v0.3.json (required)
  SFT_OUT            output JSONL (resumable; default <outputs>/sft_iterret_qasper.jsonl)
  SFT_MAX_PAPERS     papers to use (default 300; 0 = all ~888)
  SFT_WORKERS        papers processed concurrently against vLLM (default 4)
  SFT_GRAPH_CACHE    graph cache dir (default <out dir>/sft_graph_cache)
  LB_DATA            LongBench qasper.jsonl/dir -- papers found in it are EXCLUDED
"""
from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from deltamem.workmem.longctx_data import (
    format_query, load_longbench, load_qasper_raw, longbench_overlap, normalize_answer,
)
from deltamem.workmem.longctx_retrieval import (
    DEFAULT_EVIDENCE_LAYERS, DEFAULT_EVIDENCE_ORDER, evidence_items, get_or_build_doc_graph,
    make_backend, present_evidence, retrieve_evidence,
)
from iterret.llm_client import OpenAICompatibleLLMClient

_ROOT = os.environ.get("CAIMMS_WORKSPACE", os.environ.get("CAIMMS_ROOT", "."))
QASPER_TRAIN = os.environ.get("SFT_QASPER_TRAIN")
OUT = Path(os.environ.get("SFT_OUT", f"{_ROOT}/outputs/sft_iterret_qasper.jsonl"))
MAX_PAPERS = int(os.environ.get("SFT_MAX_PAPERS", "300"))
WORKERS = int(os.environ.get("SFT_WORKERS", "4"))
GRAPH_CACHE = Path(os.environ.get("SFT_GRAPH_CACHE", str(OUT.parent / "sft_graph_cache")))
LB_DATA = os.environ.get("LB_DATA") or None
VLLM_BASE_URL = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 13

_write_lock = threading.Lock()


def _done_question_ids() -> set:
    done = set()
    if OUT.exists():
        with open(OUT) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _process_paper(paper: dict, done: set, backend, stats: dict) -> None:
    llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
    pending = [qa for qa in paper["qas"] if qa["question_id"] not in done]
    if not pending:
        return
    graph, _, _ = get_or_build_doc_graph(paper["context"], f"qasper_{paper['paper_id']}", GRAPH_CACHE, llm)
    for qa in pending:
        diag: dict = {}
        try:
            evidence = retrieve_evidence(qa["question"], graph, backend, llm, diag=diag)
        except Exception as exc:  # noqa: BLE001
            print(f"[{paper['paper_id']}] IterRet failed on {qa['question_id']}: {exc}", flush=True)
            continue
        answer = qa["answers"][0]
        row = {"paper_id": paper["paper_id"], "question_id": qa["question_id"],
               "n_evidence": len(evidence), "answer": answer, "skipped": not evidence}
        if evidence:
            joined = normalize_answer(" ".join(evidence))
            # None for unanswerable / yes-no targets: they never occur verbatim.
            row["answer_in_evidence"] = (None if answer in ("unanswerable", "yes", "no")
                                         else normalize_answer(answer) in joined)
            # Relevance-sorted and UNFILTERED here; `prepare` applies the eval's
            # presentation (present_evidence) using this per-item metadata, so the
            # format can change without re-running retrieval.
            row["evidence_meta"] = [[cid, layer, pos] for _, cid, layer, pos
                                    in evidence_items(evidence, diag.get("final_evidence_ids", []), graph)]
            row["messages"] = ([{"role": "user", "content": e} for e in evidence]
                               + [{"role": "user", "content": format_query("qasper", qa["question"])},
                                  {"role": "assistant", "content": answer}])
        with _write_lock:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            stats["rows"] += 1
            stats["skipped"] += int(not evidence)
    print(f"[{paper['paper_id']}] done ({len(pending)} q) | total rows {stats['rows']}", flush=True)


def main() -> None:
    if not QASPER_TRAIN:
        raise SystemExit("[FATAL] set SFT_QASPER_TRAIN to the raw qasper-train-v0.3.json")
    papers = [p for p in load_qasper_raw(QASPER_TRAIN) if p["qas"]]
    print(f"[init] {len(papers)} train papers with answerable annotations", flush=True)
    if LB_DATA:
        overlap = longbench_overlap(papers, load_longbench("qasper", LB_DATA))
        papers = [p for p in papers if p["paper_id"] not in overlap]
        print(f"[init] excluded {len(overlap)} papers that appear in the LongBench qasper eval", flush=True)
    else:
        print("[init] WARNING: LB_DATA not set -- cannot check overlap with the LongBench eval", flush=True)
    random.Random(SEED).shuffle(papers)
    if MAX_PAPERS > 0:
        papers = papers[:MAX_PAPERS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = _done_question_ids()
    print(f"[init] {len(papers)} papers, {sum(len(p['qas']) for p in papers)} questions, "
          f"{len(done)} already done -> {OUT}", flush=True)

    backend = make_backend(thread_safe=True)
    stats = {"rows": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        futures = [pool.submit(_process_paper, p, done, backend, stats) for p in papers]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                print(f"[warn] paper failed: {exc}", flush=True)

    rows = [json.loads(line) for line in open(OUT)]
    usable = [r for r in rows if not r.get("skipped")]
    span = [r for r in usable if r.get("answer_in_evidence") is not None]
    hit = sum(1 for r in span if r["answer_in_evidence"])
    unans = sum(1 for r in usable if r["answer"] == "unanswerable")
    print("=" * 60, flush=True)
    print(f"episodes: {len(usable)} usable / {len(rows)} rows (skipped, no evidence: {len(rows) - len(usable)})")
    print(f"gold answer string found in evidence: {hit}/{len(span)} span/free-form targets")
    print(f"unanswerable targets: {unans}")
    print(f"mean evidence items: {sum(r['n_evidence'] for r in usable) / max(1, len(usable)):.1f}")
    print("=" * 60, flush=True)


def prepare(in_path: str, out_path: str, tokenizer_path: str, max_write_tokens: int,
            drop_unanswerable: bool = False, layers: str = DEFAULT_EVIDENCE_LAYERS,
            order: str = DEFAULT_EVIDENCE_ORDER) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    n_in = n_out = n_items_before = n_items_after = n_unans = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            n_in += 1
            if row.get("skipped") or not row.get("messages"):
                continue
            if row["answer"] == "unanswerable":
                n_unans += 1
                if drop_unanswerable:
                    continue
            evidence = [m["content"] for m in row["messages"][:-2]]
            meta = row.get("evidence_meta")
            if meta is None or len(meta) != len(evidence):
                raise SystemExit("[prepare] rows lack evidence_meta (built before 2026-10-09) -- "
                                 "rebuild them with build_sft_data.sh")
            items = [(t, m[0], m[1], m[2]) for t, m in zip(evidence, meta)]
            # Same function, same order of steps as eval_longbench_iterret.
            kept = present_evidence(items, layers=layers, order=order,
                                    tokenizer=tok, max_tokens=max_write_tokens)
            n_items_before += len(evidence)
            n_items_after += len(kept)
            if not kept:
                continue  # e.g. only semantic facts were retrieved
            messages = [{"role": "user", "content": it[0]} for it in kept] + row["messages"][-2:]
            fout.write(json.dumps({"messages": messages}) + "\n")
            n_out += 1
    print(f"[prepare] {n_out}/{n_in} episodes -> {out_path} | evidence items "
          f"{n_items_before / max(1, n_out):.1f} -> {n_items_after / max(1, n_out):.1f} per episode "
          f"(budget {max_write_tokens} tokens, {layers}/{order}) | unanswerable targets {n_unans}"
          f"{' (dropped)' if drop_unanswerable else ''}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--in", dest="in_path", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--tokenizer", required=True)
        ap.add_argument("--max-write-tokens", type=int, default=1536)
        ap.add_argument("--drop-unanswerable", action="store_true")
        ap.add_argument("--evidence-layers", default=DEFAULT_EVIDENCE_LAYERS, choices=["episodic", "all"])
        ap.add_argument("--evidence-order", default=DEFAULT_EVIDENCE_ORDER, choices=["document", "relevance"])
        a = ap.parse_args()
        prepare(a.in_path, a.out, a.tokenizer, a.max_write_tokens, a.drop_unanswerable,
                a.evidence_layers, a.evidence_order)
    else:
        main()
