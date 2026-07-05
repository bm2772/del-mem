"""C-AIMMS Full Pipeline: IterRet + Delta-Mem on LoCoMo."""
from __future__ import annotations
import gc, json, re, torch
from pathlib import Path
from typing import List, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.eval.locomo_protocol import score_locomo_prediction
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.iterret_bridge import get_iterret_evidence
from deltamem.workmem.osam_workmem import answer_with_osam, populate_osam_from_evidence
from deltamem.workmem.evidence_filter import filter_evidence_by_relevance
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue

MODEL_PATH   = "/home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507"
ADAPTER_DIR  = "/home/kbasu/arnavbhatt/workmem_test/models/delta-mem-adapter"
DATA_FILE    = "/home/kbasu/arnavbhatt/workmem_test/workmem-vertical/delta-Mem/data/locomo10.json"
OUTPUT_FILE  = "/home/kbasu/arnavbhatt/workmem_test/outputs/workmem_iterret_full.jsonl"
VLLM_BASE_URL   = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ITERRET_MAX_ITERATIONS = 5
MAX_SAMPLES = None



def extract_prediction(out, session) -> str:
    prediction = ""
    if isinstance(out, dict):
        prediction = (out.get("response") or out.get("assistant")
                      or out.get("text") or out.get("content") or "")
    if not prediction:
        for msg in reversed(session.messages):
            if msg.get("role") == "assistant":
                prediction = msg.get("content", "")
                break
    return prediction

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


def main() -> None:
    print(f"[init] Loading base model from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
    )
    print(f"[init] Attaching Delta-Mem adapter from {ADAPTER_DIR}", flush=True)
    attach_delta_adapter_in_place(model, Path(ADAPTER_DIR))
    model.eval()
    print("[init] Delta-Mem model ready.\n", flush=True)

    print(f"[init] Loading dataset from {DATA_FILE}", flush=True)
    with open(DATA_FILE) as f:
        samples = json.load(f)
    print(f"[init] {len(samples)} samples loaded.", flush=True)

    completed, results = load_checkpoint(OUTPUT_FILE)
    if completed:
        print(f"[checkpoint] Resuming — {len(completed)} questions already done.", flush=True)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

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
                turns.append(DialogueTurn(speaker=speaker, text=temporal_text, time=str(session_ts)))

        if not turns:
            print(f"[sample {sample_idx}] no turns, skipping.", flush=True)
            continue

        print(f"[sample {sample_idx}] Building CTC graph ({len(turns)} turns, {len(questions)} questions)...", flush=True)
        graph = None
        bank  = None
        try:
            graph = build_ctc_graph_from_dialogue(turns, graph_llm)
            n_nodes = len(graph.nodes) if hasattr(graph, "nodes") else "?"
            print(f"[sample {sample_idx}] Graph ready: {n_nodes} nodes.", flush=True)
            BANK_PATH = "/home/kbasu/arnavbhatt/workmem_test/outputs/experience_bank.json"
            if Path(BANK_PATH).exists():
                bank = ExperienceBank.load(BANK_PATH, build_default_embedding_backend())
                print(f"[sample {sample_idx}] Loaded experience bank: "
                      f"{len(bank.planning_bank)} planning, "
                      f"{len(bank.reflection_bank)} reflection entries", flush=True)
            else:
                bank = ExperienceBank(build_default_embedding_backend())
                print(f"[sample {sample_idx}] No experience bank found, using empty bank", flush=True)
        except Exception as exc:
            print(f"[sample {sample_idx}] Graph build FAILED: {exc}", flush=True)
            with open(OUTPUT_FILE, "a") as cf:
                for q_idx, question in enumerate(questions):
                    if (sample_idx, q_idx) in completed:
                        continue
                    entry = {
                        "sample_idx": sample_idx, "q_idx": q_idx,
                        "question": question.get("question", ""), "gold_answer": gold_answer_of(question),
                        "category": question.get("category"), "n_evidence_retrieved": 0,
                        "prediction": "", "score": 0.0, "skipped": True, "reason": "graph_build_failed",
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
            evidence: List[str] = []
            try:
                evidence = get_iterret_evidence(q_text, graph, bank, question_llm, max_iterations=5)
            except Exception as exc:
                print(f"[sample {sample_idx}.{q_idx}] IterRet FAILED: {exc}", flush=True)

            # --- evidence filter before OSAM write ---
            if evidence:
                evidence = filter_evidence_by_relevance(
                    q_text, evidence, bank.backend.encode, threshold=0.30
                )
                print(f'[sample {sample_idx}.{q_idx}] evidence after filter: {len(evidence)} strings', flush=True)
            # --- end filter ---
            n_ev = len(evidence)

            if not evidence:
                entry = {
                    "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                    "gold_answer": gold_answer_of(question), "category": question.get("category"),
                    "n_evidence_retrieved": 0, "prediction": "", "score": 0.0,
                    "skipped": True, "reason": "no_relevant_evidence",
                }
                with open(OUTPUT_FILE, "a") as cf:
                    cf.write(json.dumps(entry) + "\n")
                results.append(entry)
                completed.add((sample_idx, q_idx))
                print(f"[sample {sample_idx}.{q_idx}] No evidence, score=0", flush=True)
                continue

            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            session.reset()
            populate_osam_from_evidence(session, evidence)

            prediction = ""
            try:
                if cat_int == 5:
                    cat5_query = (
                        q_text
                        + " Answer ONLY with one of these two options exactly as written: "
                        "'No information available' or the specific answer from the conversation. "
                        "Do not write any other text."
                    )
                    out = answer_with_osam(session, cat5_query)
                elif cat_int == 2:
                    temporal_query = (
                        q_text
                        + " IMPORTANT: The evidence contains timestamps in the format "
                        "[YYYY-MM-DD | Session N]. You MUST use these timestamps to convert "
                        "any relative date expressions (yesterday, last week, last month, "
                        "this year, recently, etc.) into exact absolute dates or date ranges. "
                        "Answer with a specific date or time period, not a relative expression."
                    )
                    out = answer_with_osam(session, temporal_query)
                else:
                    out = answer_with_osam(session, q_text)
                prediction = extract_prediction(out, session)
            except Exception as exc:
                print(f"[sample {sample_idx}.{q_idx}] Generation FAILED: {exc}", flush=True)

            score = score_locomo_prediction(question, prediction)
            entry = {
                "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                "gold_answer": gold_answer_of(question), "category": question.get("category"),
                "n_evidence_retrieved": n_ev, "prediction": prediction, "score": score, "skipped": False,
            }
            with open(OUTPUT_FILE, "a") as cf:
                cf.write(json.dumps(entry) + "\n")
            results.append(entry)
            completed.add((sample_idx, q_idx))
            print(f"[sample {sample_idx}.{q_idx}] score={score:.3f} n_ev={n_ev} pred={prediction[:60]!r}", flush=True)

            del session
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
    print(f"OVERALL F1 (all {len(all_r)} incl. skipped): {avg_all:.4f}", flush=True)
    print(f"OVERALL F1 (answered only, {len(answered)}):  {avg_answered:.4f}", flush=True)
    CATEGORY_MAP = {1: "MULTI_HOP", 2: "TEMPORAL", 3: "OPEN_DOMAIN", 4: "SINGLE_HOP", 5: "ADVERSARIAL"}
    cat_scores: dict = {}
    for r in all_r:
        try:
            cat = int(r.get("category"))
            cat_scores.setdefault(cat, []).append(r["score"])
        except (TypeError, ValueError):
            continue
    for cat_id, cat_name in sorted(CATEGORY_MAP.items()):
        if cat_id in cat_scores:
            sc = cat_scores[cat_id]
            print(f"  [{cat_name}] {sum(sc)/len(sc):.4f}  ({len(sc)} questions)", flush=True)


if __name__ == "__main__":
    main()
