"""Paired comparison of two result files on the questions both answered.

  python3 scripts/compare_runs.py A.jsonl B.jsonl            # B - A
  python3 scripts/compare_runs.py A.jsonl B.jsonl --by category

Works for LoCoMo rows (key sample_idx/q_idx, metric "score", optional
"judge_correct") and MemoryAgentBench rows (key ctx_idx/q_idx, metrics
"substring_exact_match", "f1", optional "judge_correct"). Skipped rows and
failed generations (empty prediction) are left out. Reports per metric: mean A,
mean B, difference with a 95% paired-bootstrap interval, and how many questions
each side wins.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

LOCOMO_CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def load(path: str) -> dict:
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("skipped") or not str(r.get("prediction", "")).strip():
            continue
        key = (r["sample_idx"], r["q_idx"]) if "sample_idx" in r else (r["ctx_idx"], r["q_idx"])
        rows[key] = r  # last write wins (resumed runs)
    return rows


def metric_names(r: dict) -> list:
    names = [m for m in ("score", "substring_exact_match", "f1") if m in r]
    return names + (["judge_correct"] if "judge_correct" in r else [])


def bootstrap(diffs: list, n: int = 5000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def report(label: str, keys: list, a: dict, b: dict) -> None:
    metrics = [m for m in metric_names(a[keys[0]]) if m in b[keys[0]]]
    print(f"-- {label}  (n={len(keys)})")
    for m in metrics:
        xa = [float(a[k][m]) for k in keys]
        xb = [float(b[k][m]) for k in keys]
        d = [y - x for x, y in zip(xa, xb)]
        lo, hi = bootstrap(d) if len(d) > 1 else (0.0, 0.0)
        wins_b = sum(v > 0 for v in d)
        wins_a = sum(v < 0 for v in d)
        print(f"   {m:22} A {sum(xa) / len(xa):.4f}   B {sum(xb) / len(xb):.4f}   "
              f"B-A {sum(d) / len(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]   B better {wins_b} / A better {wins_a}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--by", default="", help="row field to break down by, e.g. category or question_type")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    print(f"A = {args.a}  ({len(a)} answered)\nB = {args.b}  ({len(b)} answered)\npaired: {len(keys)}")
    if not keys:
        return
    report("all", keys, a, b)
    if args.by:
        groups = defaultdict(list)
        for k in keys:
            g = a[k].get(args.by)
            groups[LOCOMO_CATEGORIES.get(g, g) if args.by == "category" else g].append(k)
        for g in sorted(groups, key=str):
            report(f"{args.by}={g}", groups[g], a, b)


if __name__ == "__main__":
    main()
