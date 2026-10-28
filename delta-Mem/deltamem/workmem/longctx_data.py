"""Long-context benchmark data for the IterRet + delta-mem pipeline.

* LongBench QA tasks (THUDM/LongBench): one document + one question per row.
  Loaded from a local ``<task>.jsonl`` (``fetch_longbench``) or LongBench's
  ``data.zip`` on the Hub.
* Raw Qasper (allenai, ``qasper-train-v0.3.json``): the TRAINING split, used to
  build IterRet-evidence SFT episodes for retraining delta-mem
  (``build_iterret_sft_data``). Papers that also appear in the LongBench Qasper
  eval are excluded there (``longbench_overlap``).

Prompts are LongBench's official templates (config/dataset2prompt.json), split
at ``{context}``: the context goes into the session as messages (written into S
and/or placed in the prompt, depending on the OSAM mode) and ``QUERY_TEMPLATES``
is the instruction+question block of the final user turn. Metric is LongBench's
``qa_f1_score`` (SQuAD-style normalisation, max over gold answers).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

LONGBENCH_QA_TASKS = ["qasper", "narrativeqa", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]

_QASPER_RULE = (
    "Answer the question based on the above article as concisely as you can, using a single "
    "phrase or sentence if possible. If the question cannot be answered based on the information "
    "in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", "
    "\"no\", or \"unanswerable\". Do not provide any explanation."
)
_PASSAGES_RULE = ("Answer the question based on the given passages. Only give me the answer and "
                  "do not output any other words.")

# The part of each official LongBench template AFTER "{context}".
QUERY_TEMPLATES: Dict[str, str] = {
    "qasper": _QASPER_RULE + "\n\nQuestion: {input}\n\nAnswer:",
    "narrativeqa": ("Now, answer the question based on the story as concisely as you can, using a "
                    "single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"),
    "multifieldqa_en": ("Now, answer the following question based on the above text, only give me the "
                        "answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"),
    "hotpotqa": _PASSAGES_RULE + "\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": _PASSAGES_RULE + "\n\nQuestion: {input}\nAnswer:",
    "musique": _PASSAGES_RULE + "\n\nQuestion: {input}\nAnswer:",
}

# LongBench dataset2maxlen.json
MAX_NEW_TOKENS: Dict[str, int] = {
    "qasper": 128, "narrativeqa": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
}


def format_query(task: str, question: str) -> str:
    return QUERY_TEMPLATES[task].format(input=question)


def doc_key(context: str) -> str:
    """Stable id for a document (graph-cache key): LongBench rows carry no paper id."""
    return hashlib.md5(context.encode("utf-8")).hexdigest()[:16]


# ── LongBench ────────────────────────────────────────────────────────────────

def _normalize_lb_row(raw: dict, task: str, idx: int) -> dict:
    return {
        "idx": idx,
        "id": raw.get("_id") or raw.get("id") or str(idx),
        "task": task,
        "context": raw.get("context", ""),
        "question": raw.get("input", raw.get("question", "")),
        "answers": [str(a) for a in raw.get("answers", []) if a is not None],
        "length": raw.get("length"),
    }


def load_longbench(task: str, path: Optional[str] = None) -> List[dict]:
    """``path``: a ``<task>.jsonl`` file or a directory holding one; None = Hub data.zip."""
    if path:
        jsonl = path if os.path.isfile(path) else os.path.join(path, f"{task}.jsonl")
        if not os.path.isfile(jsonl):
            raise FileNotFoundError(f"{jsonl} not found -- run deltamem.workmem.longctx_data fetch first")
        with open(jsonl, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        rows = _longbench_rows_from_hub(task)
    return [_normalize_lb_row(r, task, i) for i, r in enumerate(rows)]


def _longbench_rows_from_hub(task: str) -> List[dict]:
    # datasets>=3 refuses THUDM/LongBench's loading script; read the raw zip instead.
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl"]
        if not members:
            raise FileNotFoundError(f"{task}.jsonl not in LongBench data.zip")
        with zf.open(members[0]) as fh:
            return [json.loads(line) for line in fh.read().decode("utf-8").splitlines() if line.strip()]


def fetch_longbench(out_dir: str, tasks: Iterable[str] = ("qasper",)) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for task in tasks:
        rows = _longbench_rows_from_hub(task)
        out = os.path.join(out_dir, f"{task}.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[fetch] {task}: {len(rows)} rows -> {out}")


# ── Raw Qasper (training split) ──────────────────────────────────────────────

def qasper_paper_text(paper: dict) -> str:
    parts = [paper.get("title", "").strip(), "Abstract\n" + paper.get("abstract", "").strip()]
    for section in paper.get("full_text") or []:
        name = (section.get("section_name") or "").strip()
        paras = [p.strip() for p in section.get("paragraphs") or [] if p and p.strip()]
        if not paras:
            continue
        parts.append((name + "\n" if name else "") + "\n\n".join(paras))
    return "\n\n".join(p for p in parts if p)


def qasper_answer_string(answer: dict) -> Optional[str]:
    """One annotator's answer in LongBench's convention."""
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("extractive_spans"):
        return ", ".join(s.strip() for s in answer["extractive_spans"] if s.strip()) or None
    if answer.get("free_form_answer", "").strip():
        return answer["free_form_answer"].strip()
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    return None


def load_qasper_raw(path: str) -> List[dict]:
    """Papers from a raw Qasper JSON ({paper_id: paper}). Each paper:
    {paper_id, context, qas: [{question_id, question, answers: [str, ...]}]}."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    papers = []
    for pid, paper in data.items():
        qas = []
        for qa in paper.get("qas") or []:
            answers = [a for a in (qasper_answer_string(x.get("answer") or {})
                                   for x in qa.get("answers") or []) if a]
            if answers:
                qas.append({"question_id": qa.get("question_id"), "question": qa["question"],
                            "answers": answers})
        papers.append({"paper_id": pid, "title": paper.get("title", ""),
                       "abstract": paper.get("abstract", ""),
                       "context": qasper_paper_text(paper), "qas": qas})
    return papers


def _norm_ws(s: str) -> str:
    return " ".join(s.lower().split())


def longbench_overlap(papers: Sequence[dict], lb_rows: Sequence[dict]) -> set:
    """paper_ids whose abstract (first 200 normalised chars) or title occurs in
    any LongBench context -- these must not be used for training."""
    contexts = [_norm_ws(r["context"]) for r in lb_rows]
    hits = set()
    for p in papers:
        probes = [x for x in (_norm_ws(p.get("abstract", ""))[:200], _norm_ws(p.get("title", ""))) if len(x) > 20]
        if any(probe in ctx for probe in probes for ctx in contexts):
            hits.add(p["paper_id"])
    return hits


# ── Metric (LongBench qa_f1_score) ───────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(prediction: str, ground_truth: str) -> float:
    pred, gold = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(gold)
    return 2 * p * r / (p + r)


def qa_f1_score(prediction: str, answers: Sequence[str]) -> float:
    # Official LongBench eval.py truncates to the first line only for
    # trec/triviaqa/samsum/lsht -- not for these QA tasks -- so score as is.
    return max((_f1(prediction, a) for a in answers if a), default=0.0)


# ── MemoryAgentBench (ai-hyz/MemoryAgentBench) ───────────────────────────────
# Accurate_Retrieval split; one row = one long history + many questions about it.
# "longmemeval_s*" = LongMemEval (chat-assistant memory) packed into 5 histories.
MAB_REPO = "ai-hyz/MemoryAgentBench"
MAB_SPLIT_FILE = {"Accurate_Retrieval": "data/Accurate_Retrieval-00000-of-00001.parquet"}

# The benchmark's own long-context-agent query template for LongMemEval
# (deltamem/eval/official_memory_agent_bench_templates.py).
MAB_QUERY_TEMPLATES: Dict[str, str] = {
    "longmemeval": ("The history chats are between you and a user. Based on the relevant chat history, "
                    "answer the question as concisely as you can, using a single phrase if possible.\n\n "
                    "{question} \n\n Answer:"),
}
MAB_MAX_NEW_TOKENS = {"longmemeval": 50}


def mab_family(source: str) -> str:
    return "longmemeval" if source.startswith("longmemeval") else source


def load_mab(source: str = "longmemeval_s*", split: str = "Accurate_Retrieval",
             cache_dir: Optional[str] = None) -> List[dict]:
    """Rows of one MemoryAgentBench source, as
    {ctx_idx, context, questions: [{q_idx, question, answers, question_type, question_date}]}.
    Cached as JSONL in ``cache_dir`` after the first download."""
    safe = source.replace("*", "star")
    cache = os.path.join(cache_dir, f"{split}_{safe}.jsonl") if cache_dir else None
    if cache and os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=MAB_REPO, repo_type="dataset", filename=MAB_SPLIT_FILE[split])
    table = pq.read_table(path).to_pylist()
    rows = []
    for raw in table:
        meta = dict(raw.get("metadata") or {})
        if meta.get("source", "") != source:
            continue
        def _meta(key, i):
            v = meta.get(key)
            return v[i] if isinstance(v, list) and i < len(v) else None
        qs = []
        for i, q in enumerate(raw.get("questions") or []):
            ans = (raw.get("answers") or [])[i] if i < len(raw.get("answers") or []) else ""
            answers = [str(a) for a in ans] if isinstance(ans, (list, tuple)) else [str(ans)]
            qs.append({"q_idx": i, "question": str(q), "answers": answers,
                       "question_type": _meta("question_types", i), "question_date": _meta("question_dates", i)})
        rows.append({"ctx_idx": len(rows), "source": source, "context": str(raw.get("context", "")),
                     "questions": qs})
    if not rows:
        raise SystemExit(f"[FATAL] no MemoryAgentBench rows with metadata.source == {source!r}")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return rows


def longmemeval_passages(context: str, *, target_words: int = 180, max_words: int = 300) -> Optional[List[str]]:
    """Session-aware passages for a LongMemEval history.

    The history is the Python repr of a list alternating 'Chat Time: <date>'
    strings and lists of {'role', 'content'} turns. Turns are grouped into
    ~target_words passages WITHIN a session, and every passage is prefixed with
    its session's date -- otherwise only the first passage of a session would
    know when it happened, and temporal questions ("how many weeks ago...")
    become unanswerable from any later passage. Returns None if the history is
    not in that format (callers then fall back to plain chunking).
    """
    import ast

    try:
        data = ast.literal_eval(context)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(data, list):
        return None
    passages: List[str] = []
    when = ""
    for item in data:
        if isinstance(item, str):
            when = item.split(":", 1)[1].strip() if item.lower().startswith("chat time") else when
            continue
        if not isinstance(item, list):
            continue
        prefix = f"[{when}] " if when else ""
        buf: List[str] = []
        n = 0
        for turn in item:
            if not isinstance(turn, dict):
                continue
            words = f"{turn.get('role', '?')}: {turn.get('content', '')}".split()
            for start in range(0, len(words), max_words):  # very long turns split
                piece = " ".join(words[start:start + max_words])
                k = len(piece.split())
                if buf and n + k > max_words:
                    passages.append(prefix + "\n".join(buf))
                    buf, n = [], 0
                buf.append(piece)
                n += k
                if n >= target_words:
                    passages.append(prefix + "\n".join(buf))
                    buf, n = [], 0
        if buf:
            passages.append(prefix + "\n".join(buf))
    return passages or None


def mab_question_core(question: str) -> str:
    """The question without LongMemEval's 'Current Date: ... Now Answer the
    Question:' header -- used as the RETRIEVAL query (the prompt keeps the date)."""
    marker = "Now Answer the Question:"
    return question.split(marker, 1)[1].strip() if marker in question else question


_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def mab_question_date(question: str) -> Optional[tuple]:
    """(year, month, day) from the question's 'Current Date: YYYY/MM/DD (Day) HH:MM' header."""
    m = re.search(r"Current Date:\s*(\d{4})/(\d{2})/(\d{2})", question)
    return tuple(int(g) for g in m.groups()) if m else None


def passage_date(passage: str) -> Optional[tuple]:
    """(year, month, day) of a dated session passage ('[YYYY/MM/DD (Day) HH:MM] ...')."""
    m = _DATE_RE.match(passage[1:]) if passage.startswith("[") else None
    return tuple(int(g) for g in m.groups()) if m else None


def passages_before(passages: Sequence[str], cutoff: tuple) -> Optional[List[str]]:
    """Passages whose session date is <= cutoff, in history order. None if any
    passage has no parsable date (the caller then falls back to the history's end)."""
    dates = [passage_date(p) for p in passages]
    if any(d is None for d in dates):
        return None
    return [p for p, d in zip(passages, dates) if d <= cutoff]


def mab_scores(prediction: str, answers: Sequence[str]) -> Dict[str, float]:
    """Exact match, substring exact match (normalised gold inside normalised
    prediction -- MemoryAgentBench's accuracy for Accurate Retrieval) and token F1;
    each the max over gold answers."""
    p = normalize_answer(prediction)
    golds = [normalize_answer(a) for a in answers if str(a).strip()]
    if not golds:
        return {"exact_match": 0.0, "substring_exact_match": 0.0, "f1": 0.0}
    return {
        "exact_match": float(any(p == g for g in golds)),
        "substring_exact_match": float(any(g in p for g in golds)),
        "f1": max(_f1(prediction, a) for a in answers if str(a).strip()),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download LongBench QA tasks to local JSONL.")
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="*", default=["qasper"])
    a = ap.parse_args()
    fetch_longbench(a.out_dir, a.tasks)
