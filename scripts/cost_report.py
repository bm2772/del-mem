"""Cost report for IterRet runs (test-plan D3): per-question retrieval cost and
one-off memory-construction cost, from what the evals now record.

  python3 scripts/cost_report.py --results run1.jsonl run2.jsonl --graphs outputs/graph_cache

Per question (from each row's retrieval diagnostics): LLM calls, prompt and
completion tokens, retrieval seconds, rounds -- mean / median / p90.
Per graph (from graph_cache/*.json "meta.build_cost"): LLM calls, tokens,
seconds and units (turns/passages) to build the memory.
Rows written before cost tracking existed have no "cost" field and are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path


def _stats(xs):
    if not xs:
        return "n/a"
    xs = sorted(xs)
    p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return f"mean {st.mean(xs):9.1f}   median {st.median(xs):9.1f}   p90 {p90:9.1f}"


def report_results(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    diags = [r.get("retrieval") or {} for r in rows]
    with_cost = [d for d in diags if d.get("cost")]
    print(f"== {Path(path).name}: {len(rows)} rows, {len(with_cost)} with cost data")
    if not with_cost:
        return
    retr = {d.get("retriever", "iterret") for d in with_cost}
    print(f"   retriever(s): {sorted(retr)}")
    for key, label in (("llm_calls", "LLM calls / question"),
                       ("prompt_tokens", "prompt tokens / question"),
                       ("completion_tokens", "completion tokens / question")):
        print(f"   {label:30} {_stats([d['cost'].get(key, 0) for d in with_cost])}")
    print(f"   {'retrieval seconds / question':30} {_stats([d.get('retrieval_s', 0) for d in with_cost])}")
    print(f"   {'retrieval rounds / question':30} {_stats([d.get('rounds', 0) for d in with_cost])}")


def report_graphs(pattern_dir: str) -> None:
    files = sorted(glob.glob(str(Path(pattern_dir) / "*.json")))
    costs = []
    for f in files:
        meta = json.load(open(f)).get("meta") or {}
        if meta.get("build_cost"):
            costs.append(meta["build_cost"])
    print(f"== graphs in {pattern_dir}: {len(files)} files, {len(costs)} with build cost")
    if not costs:
        return
    units_key = "n_turns" if "n_turns" in costs[0] else "n_spans"
    for key, label in (("llm_calls", "LLM calls / graph"), ("prompt_tokens", "prompt tokens / graph"),
                       ("completion_tokens", "completion tokens / graph"), ("seconds", "seconds / graph"),
                       (units_key, "memory units / graph")):
        print(f"   {label:30} {_stats([c.get(key, 0) for c in costs])}")
    per_unit = [c["llm_calls"] / c[units_key] for c in costs if c.get(units_key)]
    print(f"   {'LLM calls per memory unit':30} {_stats(per_unit)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--graphs", nargs="*", default=[])
    a = ap.parse_args()
    for r in a.results:
        report_results(r)
    for g in a.graphs:
        report_graphs(g)


if __name__ == "__main__":
    main()
