"""C-AIMMS Ablation: IterRet + direct generation (NO Delta-Mem OSAM).
Purpose: isolate Delta-Mem's contribution by comparing:
  - This script: IterRet retrieval -> evidence as plain context -> base Qwen3-4B generates
  - eval_locomo_iterret_mock.py: same retrieval -> OSAM compression -> Delta-Mem generates

If this script scores higher, OSAM compression is losing information.
If this script scores lower, Delta-Mem is adding value beyond plain context.
"""
from __future__ import annotations
import gc, json, os, re, torch
from pathlib import Path
from typing import List, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.eval.locomo_protocol import (
    CATEGORY_DISPLAY_NAMES,
    score_locomo_prediction,
)
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue
from deltamem.workmem.iterret_bridge import get_iterret_evidence

# ── configuration ─────────────────────────────────────────────────────────────
# Paths resolve from CAIMMS_ROOT so this file runs unchanged on the A100 SLURM
# cluster (defaults below are the original absolute paths) and on any other box
# that exports CAIMMS_ROOT -- see env.sh at the bundle root.
_ROOT = os.environ.get("CAIMMS_ROOT", "/home/kbasu/arnavbhatt/workmem_test")
MODEL_PATH   = os.environ.get("CAIMMS_MODEL_PATH", f"{_ROOT}/models/Qwen3-4B-Instruct-2507")
# NO ADAPTER — base model only
DATA_FILE    = os.environ.get("CAIMMS_DATA_FILE",  f"{_ROOT}/workmem-vertical/delta-Mem/data/locomo10.json")
OUTPUT_FILE  = os.environ.get("CAIMMS_ABLATION_OUTPUT", f"{_ROOT}/outputs/workmem_ablation_direct.jsonl")
VLLM_BASE_URL   = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ITERRET_MAX_ITERATIONS = 5
MAX_SAMPLES  = None
MAX_EVIDENCE_TOKENS = 2048  # cap evidence to avoid context overflow
# ──────────────────────────────────────────────────────────────────────────────


def extract_session_nums(conv_block: dict) -> List[int]:
    nums = []
    for k in conv_block:
        m = re.fullmatch(r"session_(\d+)", k)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def load_checkpoint(output_file: str) -> Tuple[Set[Tuple[int, int]], List[dict]]:
    completed: Set[Tuple[int, int]] = set()
    results: List[dict] = []
    path = Path(output_file)
    if not path.exists():
        return completed, results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed.add((entry["sample_idx"], entry["q_idx"]))
                results.append(entry)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed, results


def gold_answer_of(q: dict) -> str:
    ans = q.get("answer", "")
    if isinstance(ans, list):
        return ans[0] if ans else ""
    return str(ans)


def build_evidence_context(evidence: List[str], tokenizer, max_tokens: int) -> str:
    """Concatenate evidence strings into a single context block.
    Truncates from the front if total exceeds max_tokens, keeping
    most recent evidence (which is typically most relevant).
    """
    joined = "\n\n".join(f"[Evidence {i+1}] {e}" for i, e in enumerate(evidence))
    tokens = tokenizer.encode(joined)
    if len(tokens) > max_tokens:
        # truncate from front — keep the end (most recent retrieval iterations)
        tokens = tokens[-max_tokens:]
        joined = tokenizer.decode(tokens, skip_special_tokens=True)
    return joined


def generate_direct(
    model,
    tokenizer,
    evidence_context: str,
    query: str,
    cat_int: int,
    device: str = "cuda:0",
) -> str:
    """Generate answer directly from evidence context — no OSAM, no adapter."""
    if cat_int == 5:
        prompt_text = (
            f"Evidence from conversation:\n{evidence_context}\n\n"
            f"Question: {query}\n"
            f"Answer ONLY with one of these two options exactly as written: "
            f"'No information available' or the specific answer from the conversation. "
            f"Do not write any other text.\nAnswer:"
        )
    elif cat_int == 2:
        prompt_text = (
            f"Evidence from conversation:\n{evidence_context}\n\n"
            f"Question: {query}\n"
            f"IMPORTANT: Use the timestamps in the evidence to convert any relative dates "
            f"(yesterday, last week, etc.) into exact absolute dates. "
            f"Answer with a specific date or time period in third person.\nAnswer:"
        )
    else:
        prompt_text = (
            f"Evidence from conversation:\n{evidence_context}\n\n"
            f"Based on the evidence above, answer the following question in third person "
            f"using exact words from the evidence where possible. "
            f"Answer in a few words only. "
            f"Do NOT answer as one of the characters.\n"
            f"Question: {query}\nAnswer:"
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about conversations accurately and concisely."},
        {"role": "user", "content": prompt_text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    # truncate if over limit
    if input_ids.shape[1] > 7000:
        input_ids = input_ids[:, -7000:]

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][input_ids.shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # take first line only — same canonicalization as Delta-Mem pipeline
    lines = [l.strip() for l in prediction.splitlines() if l.strip()]
    return lines[0] if lines else prediction


def main() -> None:
    # Load BASE model only — no adapter
    print(f"[init] Loading BASE model (no adapter) from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    print("[init] Base model ready (NO Delta-Mem adapter).\n", flush=True)

    print(f"[init] Loading dataset from {DATA_FILE}", flush=True)
    with open(DATA_FILE) as f:
        samples = json.load(f)
    print(f"[init] {len(samples)} samples loaded.", flush=True)

    completed, results = load_checkpoint(OUTPUT_FILE)
    if completed:
        print(f"[checkpoint] Resuming — {len(completed)} questions already done.", flush=True)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Experience bank — load if exists, empty otherwise
    BANK_PATH = os.environ.get("CAIMMS_BANK_PATH", f"{_ROOT}/outputs/experience_bank.json")

    graph_llm    = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
    question_llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)

    for sample_idx, sample in enumerate(samples):
        if MAX_SAMPLES is not None and sample_idx >= MAX_SAMPLES:
            break

        torch.cuda.empty_cache()
        gc.collect()

        conv_block = sample.get("conversation", {})
        questions  = sample.get("qa", [])
        if not questions:
            continue

        all_done = all((sample_idx, qi) in completed for qi in range(len(questions)))
        if all_done:
            print(f"[sample {sample_idx}] fully checkpointed, skipping.", flush=True)
            continue

        # Build turns with timestamps
        turns: List[DialogueTurn] = []
        session_nums = extract_session_nums(conv_block)
        for session_num in session_nums:
            session_key = f"session_{session_num}"
            dt_key      = f"session_{session_num}_date_time"
            session_ts  = conv_block.get(dt_key, f"Session {session_num}")
            for dialog in conv_block.get(session_key, []):
                speaker = dialog.get("speaker", "Unknown")
                text    = dialog.get("text", "")
                temporal_text = f"[{session_ts}] {speaker}: {text}"
                turns.append(DialogueTurn(
                    speaker=speaker, text=temporal_text, time=str(session_ts)
                ))

        if not turns:
            print(f"[sample {sample_idx}] no turns, skipping.", flush=True)
            continue

        print(
            f"[sample {sample_idx}] Building CTC graph "
            f"({len(turns)} turns, {len(questions)} questions)...",
            flush=True,
        )
        graph = None
        bank  = None
        try:
            graph = build_ctc_graph_from_dialogue(turns, graph_llm)
            n_nodes = len(graph.nodes) if hasattr(graph, "nodes") else "?"
            print(f"[sample {sample_idx}] Graph ready: {n_nodes} nodes.", flush=True)
            if Path(BANK_PATH).exists():
                bank = ExperienceBank.load(BANK_PATH, build_default_embedding_backend())
                print(
                    f"[sample {sample_idx}] Loaded experience bank: "
                    f"{len(bank.planning_bank)} planning, "
                    f"{len(bank.reflection_bank)} reflection entries",
                    flush=True,
                )
            else:
                bank = ExperienceBank(build_default_embedding_backend())
                print(f"[sample {sample_idx}] No experience bank, using empty.", flush=True)
        except Exception as exc:
            print(f"[sample {sample_idx}] Graph build FAILED: {exc}", flush=True)
            with open(OUTPUT_FILE, "a") as cf:
                for q_idx, question in enumerate(questions):
                    if (sample_idx, q_idx) in completed:
                        continue
                    entry = {
                        "sample_idx": sample_idx, "q_idx": q_idx,
                        "question": question.get("question", ""),
                        "gold_answer": gold_answer_of(question),
                        "category": question.get("category"),
                        "n_evidence_retrieved": 0,
                        "prediction": "", "score": 0.0,
                        "skipped": True, "reason": "graph_build_failed",
                    }
                    cf.write(json.dumps(entry) + "\n")
                    results.append(entry)
                    completed.add((sample_idx, q_idx))
            continue

        for q_idx, question in enumerate(questions):
            if (sample_idx, q_idx) in completed:
                continue

            q_text = question.get("question", "")
            try:
                cat_int = int(question.get("category", 0))
            except (TypeError, ValueError):
                cat_int = 0

            # IterRet retrieval — identical to OSAM pipeline
            evidence: List[str] = []
            try:
                evidence = get_iterret_evidence(
                    q_text, graph, bank, question_llm,
                    max_iterations=ITERRET_MAX_ITERATIONS,
                )
            except Exception as exc:
                print(
                    f"[sample {sample_idx}.{q_idx}] IterRet FAILED: {exc}",
                    flush=True,
                )

            n_ev = len(evidence)

            if not evidence:
                entry = {
                    "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                    "gold_answer": gold_answer_of(question),
                    "category": question.get("category"),
                    "n_evidence_retrieved": 0, "prediction": "", "score": 0.0,
                    "skipped": True, "reason": "no_relevant_evidence",
                }
                with open(OUTPUT_FILE, "a") as cf:
                    cf.write(json.dumps(entry) + "\n")
                results.append(entry)
                completed.add((sample_idx, q_idx))
                print(
                    f"[sample {sample_idx}.{q_idx}] No evidence, score=0",
                    flush=True,
                )
                continue

            # Direct generation — evidence as plain text context, no OSAM
            evidence_context = build_evidence_context(
                evidence, tokenizer, MAX_EVIDENCE_TOKENS
            )
            prediction = ""
            try:
                prediction = generate_direct(
                    model, tokenizer, evidence_context, q_text, cat_int,
                    device="cuda:0",
                )
            except Exception as exc:
                print(
                    f"[sample {sample_idx}.{q_idx}] Generation FAILED: {exc}",
                    flush=True,
                )

            score = score_locomo_prediction(question, prediction)
            entry = {
                "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                "gold_answer": gold_answer_of(question),
                "category": question.get("category"),
                "n_evidence_retrieved": n_ev,
                "prediction": prediction, "score": score, "skipped": False,
            }
            with open(OUTPUT_FILE, "a") as cf:
                cf.write(json.dumps(entry) + "\n")
            results.append(entry)
            completed.add((sample_idx, q_idx))
            print(
                f"[sample {sample_idx}.{q_idx}] score={score:.3f} "
                f"n_ev={n_ev} pred={prediction[:60]!r}",
                flush=True,
            )

            torch.cuda.empty_cache()

        del graph, bank
        torch.cuda.empty_cache()
        gc.collect()

    answered = [r for r in results if not r.get("skipped", False)]
    all_r    = results
    if not all_r:
        print("No results.", flush=True)
        return

    avg_all      = sum(r["score"] for r in all_r) / len(all_r)
    avg_answered = sum(r["score"] for r in answered) / len(answered) if answered else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"ABLATION (IterRet + direct, NO OSAM)", flush=True)
    print(f"F1 (all {len(all_r)} incl. skipped): {avg_all:.4f}", flush=True)
    print(f"F1 (answered only, {len(answered)}):  {avg_answered:.4f}", flush=True)
    cat_scores: dict = {}
    for r in all_r:
        try:
            cat = int(r.get("category"))
            cat_scores.setdefault(cat, []).append(r["score"])
        except (TypeError, ValueError):
            continue
    for cat_id, cat_name in sorted(CATEGORY_DISPLAY_NAMES.items()):
        if cat_id in cat_scores:
            sc = cat_scores[cat_id]
            print(
                f"  [{cat_name}] {sum(sc)/len(sc):.4f}  ({len(sc)} questions)",
                flush=True,
            )


if __name__ == "__main__":
    main()
