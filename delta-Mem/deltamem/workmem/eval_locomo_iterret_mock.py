"""C-AIMMS Full Pipeline: IterRet + Delta-Mem on LoCoMo."""
from __future__ import annotations
import gc, json, os, re, torch
from pathlib import Path
from typing import List, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.eval.locomo_protocol import (
    ADVERSARIAL_CATEGORY,
    SCORED_CATEGORY_DISPLAY_NAMES,
    score_locomo_prediction,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.iterret_bridge import get_iterret_evidence
from deltamem.workmem.osam_workmem import (
    answer_with_osam, populate_osam_from_evidence, maybe_narrow_evidence,
)
from deltamem.workmem.evidence_filter import filter_evidence_by_relevance
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue
from iterret.ctc_graph import CueTagContentGraph

# Paths resolve from CAIMMS_ROOT so this file runs unchanged on the A100 SLURM
# cluster (defaults below are the original absolute paths) and on any other box
# that exports CAIMMS_ROOT -- see env.sh at the bundle root.
_ROOT = os.environ.get("CAIMMS_ROOT", "/home/kbasu/arnavbhatt/workmem_test")
MODEL_PATH   = os.environ.get("CAIMMS_MODEL_PATH",  f"{_ROOT}/models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR  = os.environ.get("CAIMMS_ADAPTER_DIR", f"{_ROOT}/models/delta-mem-adapter")
DATA_FILE    = os.environ.get("CAIMMS_DATA_FILE",   f"{_ROOT}/workmem-vertical/delta-Mem/data/locomo10.json")
# Overridable because port 8000 is not guaranteed free on a shared workstation.
VLLM_BASE_URL   = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ITERRET_MAX_ITERATIONS = 5

# Both of these are overridable from the environment so a smoke test can be run
# WITHOUT editing this file and without touching the real run's checkpoint:
#   WORKMEM_MAX_SAMPLES=1  -> process only the first N conversations
#   WORKMEM_OUTPUT_FILE=... -> write to a scratch checkpoint instead of the real one
# Keeping the output file separate matters: rows written by a smoke test would
# otherwise be picked up by the real run's checkpoint-resume and silently skipped.
OUTPUT_FILE = os.environ.get(
    "WORKMEM_OUTPUT_FILE",
    f"{_ROOT}/outputs/workmem_iterret_full.jsonl",
)
MAX_SAMPLES = int(os.environ["WORKMEM_MAX_SAMPLES"]) if os.environ.get("WORKMEM_MAX_SAMPLES") else None



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
    print(f"[init] MAX_SAMPLES={MAX_SAMPLES!r}  OUTPUT_FILE={OUTPUT_FILE!r}", flush=True)
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

    # Cache each sample's built CTC graph next to the checkpoint file (in the
    # same dir WORKMEM_OUTPUT_FILE points at, so a smoke test's cache never
    # collides with the real run's). If a job dies mid-conversation, the graph
    # for the sample that was in progress no longer has to be rebuilt from the
    # ~400 LLM calls in build_ctc_graph_from_dialogue on resume -- only the
    # remaining unanswered questions in it get (re)processed.
    GRAPH_CACHE_DIR = Path(OUTPUT_FILE).parent / "graph_cache"
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

        graph = None
        bank  = None
        cache_path = GRAPH_CACHE_DIR / f"sample_{sample_idx}.json"
        try:
            if cache_path.exists():
                print(f"[sample {sample_idx}] Loading cached CTC graph from {cache_path}...", flush=True)
                graph = CueTagContentGraph.load(str(cache_path))
            else:
                print(f"[sample {sample_idx}] Building CTC graph ({len(turns)} turns, {len(questions)} questions)...", flush=True)
                graph = build_ctc_graph_from_dialogue(turns, graph_llm)
                graph.save(str(cache_path))
            n_nodes = len(graph.nodes) if hasattr(graph, "nodes") else "?"
            print(f"[sample {sample_idx}] Graph ready: {n_nodes} nodes.", flush=True)
            # Experience bank disabled: run 3 showed loading the single-conversation
            # bank (sample 0 only) hurt multi-hop (-0.022) and temporal (-0.024)
            # vs empty bank. A bank built from 1 of 10 conversations does not
            # generalise to the other 9. Use empty bank until a diverse bank
            # built from multiple conversations is available.
            bank = ExperienceBank(build_default_embedding_backend())
            print(f"[sample {sample_idx}] Using empty experience bank (diverse bank not available).", flush=True)
            # Give the graph the same MiniLM backend the bank uses, so
            # reflect_node's fail-open fallback (nodes.py) can rank candidates
            # by embedding similarity instead of keeping everything untouched
            # when the routing LLM's decision can't be trusted.
            graph.attach_embedder(bank.backend)
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
            # Adversarial questions are excluded from this evaluation entirely
            # — no retrieval, no generation, no entry written, no score.
            if cat_int == ADVERSARIAL_CATEGORY:
                continue

            evidence: List[str] = []
            # Populated in place by get_iterret_evidence. Everything in here is
            # written to the result row below: until now a finished run stored
            # only n_evidence_retrieved, so no post-hoc analysis could tell
            # WHICH nodes were retrieved or whether routing had actually made
            # the decision -- which is why every retrieval hypothesis in the
            # handoff's loss attribution stayed unfalsifiable.
            retrieval_diag: dict = {}
            try:
                evidence = get_iterret_evidence(
                    q_text, graph, bank, question_llm,
                    max_iterations=ITERRET_MAX_ITERATIONS,
                    diag=retrieval_diag,
                )
            except Exception as exc:
                print(f"[sample {sample_idx}.{q_idx}] IterRet FAILED: {exc}", flush=True)

            # Snapshot before filtering: retrieval_diag["evidence_ids"] is
            # index-aligned with THIS list. The filter below both drops items
            # and re-sorts by relevance, so after it runs the ids can no
            # longer be matched positionally -- only by text lookup.
            pre_filter_evidence = list(evidence)

            # --- evidence filter before OSAM write ---
            if evidence:
                evidence = filter_evidence_by_relevance(
                    q_text, evidence, bank.backend.encode, threshold=0.30
                )
                # Temporal narrowing, now OFF by default -- see
                # TEMPORAL_NARROWING_ENABLED in osam_workmem.py for why. This
                # is a no-op unless WORKMEM_TEMPORAL_NARROWING=1.
                narrowed = maybe_narrow_evidence(q_text, evidence, bank.backend.encode)
                if len(narrowed) != len(evidence):
                    print(f'[sample {sample_idx}.{q_idx}] adaptive narrowing: {len(evidence)} -> {len(narrowed)}', flush=True)
                evidence = narrowed
                print(f'[sample {sample_idx}.{q_idx}] evidence after filter: {len(evidence)} strings', flush=True)
            # --- end filter ---
            n_ev = len(evidence)

            _text_to_id = dict(zip(pre_filter_evidence, retrieval_diag.get("evidence_ids", [])))
            retrieval_diag["final_evidence_ids"] = [_text_to_id.get(t, "?") for t in evidence]
            retrieval_diag["n_dropped_by_filter"] = len(pre_filter_evidence) - n_ev

            if not evidence:
                entry = {
                    "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                    "gold_answer": gold_answer_of(question), "category": question.get("category"),
                    "n_evidence_retrieved": 0, "prediction": "", "score": 0.0,
                    "skipped": True, "reason": "no_relevant_evidence",
                    "retrieval": retrieval_diag,
                }
                with open(OUTPUT_FILE, "a") as cf:
                    cf.write(json.dumps(entry) + "\n")
                results.append(entry)
                completed.add((sample_idx, q_idx))
                print(f"[sample {sample_idx}.{q_idx}] No evidence, score=0", flush=True)
                continue

            session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
            session.reset()
            # No fixed OSAM cap here (removed): it bounded total evidence
            # per question regardless of whether each item was a deliberate,
            # well-reasoned keep or fail-open noise, and penalized multi-hop
            # questions that legitimately need more than 8 items chained
            # across rounds (run 7). Capacity control now happens upstream,
            # per-round, in nodes.py's fail-open fallback (FAIL_OPEN_FALLBACK_TOP_K)
            # -- which only limits evidence from rounds where routing couldn't
            # be trusted, leaving genuinely-vetted multi-round evidence intact.
            populate_osam_from_evidence(session, evidence)

            prediction = ""
            # How much OSAM's readout contributed to the output, right at the
            # moment the model finished reading the prompt (the read that
            # produces the first generated token) -- see
            # collect_delta_mem_output_ratio_stats. {} if generation fails
            # before ever producing a result.
            osam_contribution: dict = {}
            try:
                # No category-based dispatch here -- answer_with_osam's
                # instructions are the same for every question regardless of
                # type (see its docstring/comments). cat_int is only used
                # above for the adversarial exclusion and below for score
                # bucketing.
                out = answer_with_osam(session, q_text)
                prediction = extract_prediction(out, session)
                if isinstance(out, dict):
                    osam_contribution = out.get("prompt_output_ratio_stats") or {}
            except Exception as exc:
                print(f"[sample {sample_idx}.{q_idx}] Generation FAILED: {exc}", flush=True)

            score = score_locomo_prediction(question, prediction)
            entry = {
                "sample_idx": sample_idx, "q_idx": q_idx, "question": q_text,
                "gold_answer": gold_answer_of(question), "category": question.get("category"),
                "n_evidence_retrieved": n_ev, "prediction": prediction, "score": score, "skipped": False,
                "retrieval": retrieval_diag,
                "osam_contribution": osam_contribution,
            }
            with open(OUTPUT_FILE, "a") as cf:
                cf.write(json.dumps(entry) + "\n")
            results.append(entry)
            completed.add((sample_idx, q_idx))
            ratio = osam_contribution.get("mean_delta_o_ratio")
            ratio_str = f"{ratio:.4f}" if ratio is not None else "?"
            print(f"[sample {sample_idx}.{q_idx}] score={score:.3f} n_ev={n_ev} "
                  f"osam_ratio={ratio_str} pred={prediction[:60]!r}", flush=True)

            del session
            torch.cuda.empty_cache()

        del graph, bank
        torch.cuda.empty_cache()
        gc.collect()

    # Exclude category 5 (adversarial) from every metric, including any rows
    # already checkpointed on disk from before this category was hard-excluded.
    def _category_of(r: dict) -> int:
        try:
            return int(r.get("category") or 0)
        except (TypeError, ValueError):
            return 0

    all_r    = [r for r in results if _category_of(r) != ADVERSARIAL_CATEGORY]
    answered = [r for r in all_r if not r.get("skipped", False)]
    if not all_r:
        print("No results.", flush=True)
        return
    avg_all      = sum(r["score"] for r in all_r) / len(all_r)
    avg_answered = sum(r["score"] for r in answered) / len(answered) if answered else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL F1 (all {len(all_r)} incl. skipped): {avg_all:.4f}", flush=True)
    print(f"OVERALL F1 (answered only, {len(answered)}):  {avg_answered:.4f}", flush=True)
    cat_scores: dict = {}
    for r in all_r:
        try:
            cat = int(r.get("category"))
            cat_scores.setdefault(cat, []).append(r["score"])
        except (TypeError, ValueError):
            continue
    for cat_id, cat_name in sorted(SCORED_CATEGORY_DISPLAY_NAMES.items()):
        if cat_id in cat_scores:
            sc = cat_scores[cat_id]
            print(f"  [{cat_name}] {sum(sc)/len(sc):.4f}  ({len(sc)} questions)", flush=True)


if __name__ == "__main__":
    main()
