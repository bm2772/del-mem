"""Re-score MemoryAgentBench LongMemEval results with the OFFICIAL LLM-judge
prompts, using an open-source judge served by vLLM instead of GPT-4o.

MemoryAgentBench grades LongMemEval with an LLM judge
(llm_based_eval/longmem_qa_evaluate.py in HUST-AI-HYZ/MemoryAgentBench), not
with substring match: per-question-type prompts (temporal answers forgive
off-by-one, knowledge-update accepts old+new values, preference questions are
graded against a rubric, and unanswerable "_abs" questions count as correct when
the model says the information is missing). This imports that exact prompt
builder and replicates the official call (temperature 0, max_tokens 10,
label = "yes" in the reply). Only the judge model differs (open-source).

  python -m deltamem.workmem.judge_mab_longmemeval \\
      --results a.jsonl b.jsonl --judge-url http://localhost:8010/v1 --judge-model judge \\
      --mab-root ~/caimms/external/MemoryAgentBench

Accepts the harness's full .jsonl/.json records or the slim copies in
cached_results/mab_official (fields context_id, question_index, output).
Writes <results>.judged_<tag>.jsonl next to each input and prints accuracy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_official_prompt_builder(mab_root: Path):
    """Import get_anscheck_prompt from the official script. Its module also
    imports backoff/dotenv/datasets at top level; stub whatever is missing (the
    prompt builder itself only formats strings)."""
    for name in ("backoff", "dotenv"):
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            stub.load_dotenv = lambda *a, **k: None
            stub.on_exception = lambda *a, **k: (lambda f: f)
            stub.expo = None
            sys.modules[name] = stub
    path = mab_root / "llm_based_eval" / "longmem_qa_evaluate.py"
    spec = importlib.util.spec_from_file_location("mab_longmem_qa_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_anscheck_prompt


def _load_references(source: str) -> dict:
    """(context_id, question_index) -> question, answer, question_id, question_type,
    in the same row order the harness uses (filtered by metadata.source)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="ai-hyz/MemoryAgentBench", repo_type="dataset",
                           filename="data/Accurate_Retrieval-00000-of-00001.parquet")
    refs, ctx = {}, 0
    for row in pq.read_table(path).to_pylist():
        meta = row.get("metadata") or {}
        if meta.get("source") != source:
            continue
        for i, q in enumerate(row["questions"]):
            refs[(ctx, i)] = {"question": q, "answer": row["answers"][i],
                              "question_id": meta["question_ids"][i],
                              "question_type": meta["question_types"][i]}
        ctx += 1
    return refs


def _read_results(path: Path) -> list:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)["data"]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--judge-url", default="http://localhost:8010/v1")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--tag", default="judge", help="suffix for the output file")
    ap.add_argument("--mab-root", default=str(Path.home() / "caimms/external/MemoryAgentBench"))
    ap.add_argument("--source", default="longmemeval_s*")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    from openai import OpenAI

    build_prompt = _load_official_prompt_builder(Path(a.mab_root))
    refs = _load_references(a.source)
    client = OpenAI(base_url=a.judge_url, api_key="not-needed")

    def judge(rec: dict) -> dict:
        ref = refs[(rec["context_id"], rec["question_index"])]
        # same guard as the official script: rows must align with the references
        if "answer" in rec and rec["answer"] != ref["answer"]:
            raise ValueError(f"answer mismatch at {rec['context_id']}.{rec['question_index']}: "
                             f"{rec['answer']!r} vs reference {ref['answer']!r}")
        abstention = "_abs" in ref["question_id"]
        prompt = build_prompt(ref["question_type"], ref["question"], ref["answer"], rec["output"],
                              abstention=abstention)
        reply = client.chat.completions.create(  # official settings
            model=a.judge_model, messages=[{"role": "user", "content": prompt}],
            n=1, temperature=0, max_tokens=10).choices[0].message.content.strip()
        return {"context_id": rec["context_id"], "question_index": rec["question_index"],
                "question_id": ref["question_id"], "question_type": ref["question_type"],
                "abstention": abstention, "output": rec["output"], "judge_raw": reply,
                "judge_label": "yes" in reply.lower()}

    for res in a.results:
        path = Path(res)
        recs = _read_results(path)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            judged = list(pool.map(judge, recs))
        out = path.with_name(path.name.replace(".jsonl", "").replace(".json", "") + f".judged_{a.tag}.jsonl")
        with open(out, "w") as fh:
            for j in judged:
                fh.write(json.dumps(j) + "\n")
        by_type = defaultdict(list)
        for j in judged:
            by_type["ABSTENTION" if j["abstention"] else j["question_type"]].append(j["judge_label"])
        acc = sum(j["judge_label"] for j in judged) / len(judged)
        print(f"== {path.name}: judged accuracy {100 * acc:.2f} (n={len(judged)}) -> {out.name}")
        for t, v in sorted(by_type.items()):
            print(f"   {t:28} n={len(v):3}  {100 * sum(v) / len(v):6.2f}")


if __name__ == "__main__":
    main()
