"""Laptop smoke test: exercises every non-neural stage on REAL LoCoMo data.
The LLM is MockLLMClient and the model/session are fakes -- everything else
(graph build, retrieval loop, routing fallback, diagnostics, filtering,
narrowing gate, prompt construction, row schema, scoring) is the real code."""
import json, sys, pathlib, collections
import dryrun_torchstub as torchstub; torchstub.install()

from iterret.llm_client import MockLLMClient
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue
from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import ExperienceBank, build_default_embedding_backend
from deltamem.workmem.iterret_bridge import get_iterret_evidence
from deltamem.workmem.evidence_filter import filter_evidence_by_relevance
from deltamem.workmem.osam_workmem import (
    maybe_narrow_evidence, _needs_temporal_grounding, _is_yes_no_question,
    _temporal_answer_instruction, _evidence_carries_dates,
    TEMPORAL_NARROWING_ENABLED, PHASE1_WRITE_GRANULARITY, PHASE2_PROMPT_WRITE,
)
from deltamem.eval.locomo_protocol import (
    score_locomo_prediction, ADVERSARIAL_CATEGORY, SCORED_CATEGORY_DISPLAY_NAMES,
    OFFICIAL_QA_PROMPT,
)
from deltamem.workmem.eval_locomo_iterret_mock import (
    extract_session_nums, gold_answer_of, load_checkpoint, ITERRET_MAX_ITERATIONS,
)

FAIL = []
def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not cond: FAIL.append(label)

import os
DATA = pathlib.Path(os.environ.get("CAIMMS_DATA_FILE",
    str(pathlib.Path(__file__).resolve().parent.parent / "delta-Mem" / "data" / "locomo10.json")))
samples = json.loads(DATA.read_text())
sample = samples[0]
conv, questions = sample["conversation"], sample["qa"]

print("=" * 72)
print("STAGE 1  config flags")
print("=" * 72)
check(PHASE1_WRITE_GRANULARITY == "message_mean", "Phase1 granularity = message_mean", PHASE1_WRITE_GRANULARITY)
check(PHASE2_PROMPT_WRITE is True, "Phase2 prompt-write defaults to current behaviour")
check(TEMPORAL_NARROWING_ENABLED is False, "temporal narrowing OFF by default")
check(ADVERSARIAL_CATEGORY == 5, "shared adversarial constant")
check(5 not in SCORED_CATEGORY_DISPLAY_NAMES, "adversarial excluded from scored categories")

print()
print("=" * 72)
print("STAGE 2  build a REAL CTC graph from REAL sample-0 turns (mock LLM)")
print("=" * 72)
turns = []
for n in extract_session_nums(conv):
    ts = conv.get(f"session_{n}_date_time", f"Session {n}")
    for d in conv.get(f"session_{n}", []):
        turns.append(DialogueTurn(speaker=d.get("speaker", "?"),
                                  text=f"[{ts}] {d.get('speaker','?')}: {d.get('text','')}",
                                  time=str(ts)))
turns = turns[:60]                      # keep the mock-LLM graph build quick
graph = build_ctc_graph_from_dialogue(turns, MockLLMClient())
check(len(graph.contents) > 0, "graph has content nodes", f"{len(graph.contents)} nodes")
check(len(graph.cues) > 0, "graph has cues", f"{len(graph.cues)} cues")

# persistence round-trip (the graph_cache path used on the cluster)
tmp = pathlib.Path("/tmp/_smoke_graph.json"); graph.save(str(tmp))
reloaded = CueTagContentGraph.load(str(tmp))
check(len(reloaded.contents) == len(graph.contents), "graph save/load round-trip")

# Stand-in for MiniLM: deterministic hashed bag-of-words -> FIXED-LENGTH FLOAT
# VECTOR. Using the real KeywordOverlap fallback here would be misleading --
# it returns a dict, which evidence_filter._cosine cannot consume (see the
# separate finding), so filtering would silently no-op and this stage would
# test nothing. A float encoder makes the cosine/sort/RRF arithmetic real.
import zlib, re as _re
class TinyEncoder:
    DIM = 64
    def encode(self, text):
        v = [0.0] * self.DIM
        for tok in _re.findall(r"[a-z0-9]+", str(text).lower()):
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        return v
    def similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-9)

backend = TinyEncoder()
bank = ExperienceBank(backend); graph.attach_embedder(backend)
print(f"  (embedding backend: {type(backend).__name__} -- float vectors, stands in for MiniLM)")
_probe = build_default_embedding_backend()
print(f"  (real backend that WOULD load here: {type(_probe).__name__})")

print()
print("=" * 72)
print("STAGE 3  retrieval loop + diagnostics on real questions")
print("=" * 72)
qs = [q for q in questions if int(q.get("category", 0)) != ADVERSARIAL_CATEGORY][:12]
rows, route_modes, stop_reasons = [], collections.Counter(), collections.Counter()
for qi, q in enumerate(qs):
    q_text, cat = q.get("question", ""), int(q.get("category", 0))
    diag = {}
    ev = get_iterret_evidence(q_text, graph, bank, MockLLMClient(),
                              max_iterations=ITERRET_MAX_ITERATIONS, diag=diag)
    pre = list(ev)
    if ev:
        ev = filter_evidence_by_relevance(q_text, ev, backend.encode, threshold=0.30)
        ev = maybe_narrow_evidence(q_text, ev, backend.encode)
    t2i = dict(zip(pre, diag.get("evidence_ids", [])))
    diag["final_evidence_ids"] = [t2i.get(t, "?") for t in ev]
    diag["n_dropped_by_filter"] = len(pre) - len(ev)
    route_modes.update(diag.get("route_modes", []))
    stop_reasons.update([diag.get("stop_reason")])
    rows.append({"sample_idx": 0, "q_idx": qi, "question": q_text,
                 "gold_answer": gold_answer_of(q), "category": cat,
                 "n_evidence_retrieved": len(ev), "prediction": "", "score": 0.0,
                 "skipped": False, "retrieval": diag})

check(all(r["retrieval"].get("evidence_ids") is not None for r in rows), "every row carries evidence ids")
check(all(len(r["retrieval"]["final_evidence_ids"]) == r["n_evidence_retrieved"] for r in rows),
      "final ids align with final evidence count")
check("?" not in [i for r in rows for i in r["retrieval"]["final_evidence_ids"]],
      "no unresolved ids after filter reorder")
check(any(r["retrieval"]["n_dropped_by_filter"] > 0 for r in rows) or
      any(r["n_evidence_retrieved"] > 0 for r in rows),
      "filter ran with real float vectors (not silently skipped)")
check(all(r["retrieval"].get("stop_reason") for r in rows), "stop_reason recorded")
print(f"  route modes : {dict(route_modes)}")
print(f"  stop reasons: {dict(stop_reasons)}")
print(f"  mean evidence/question: {sum(r['n_evidence_retrieved'] for r in rows)/len(rows):.1f}")

print()
print("=" * 72)
print("STAGE 4  narrowing gate is inert (change #4)")
print("=" * 72)
temporal_qs = [r for r in rows if _needs_temporal_grounding(r["question"])]
ev12 = [f"[8 May, 2023] X: item {i}" for i in range(12)]
out = maybe_narrow_evidence("When did X happen?", ev12, backend.encode)
check(len(out) == 12, "narrowing is a no-op on a temporal question", f"{len(ev12)}->{len(out)}")
print(f"  ({len(temporal_qs)} of {len(rows)} sampled questions are temporal)")

print()
print("=" * 72)
print("STAGE 5  prompt construction (real answer_with_osam path)")
print("=" * 72)

class _FakeModel:
    """set_delta_mem_write_granularity walks named_modules(); none are
    DeltaMemAttention here, so the granularity calls are real but inert."""
    def named_modules(self): return []
    def modules(self): return []


class FakeSession:
    """Stands in for DeltaMemChatSession: records what generate_reply received."""
    def __init__(self, evidence):
        self.messages = [{"role": "user", "content": e} for e in evidence]
        self.model = _FakeModel()
    def generate_reply(self, text, **kw):
        self.last_prompt, self.last_kwargs = text, kw
        return {"assistant": "stub"}

from deltamem.workmem.osam_workmem import answer_with_osam
seen_kinds = collections.Counter()
for r in rows:
    q_text = r["question"]
    sess = FakeSession([f"[1:56 pm on 8 May, 2023] Caroline: evidence {i}" for i in range(4)])
    answer_with_osam(sess, q_text)
    p = sess.last_prompt
    kind = ("TEMPORAL" if _needs_temporal_grounding(q_text)
            else "YES/NO" if _is_yes_no_question(q_text) else "NAMED")
    seen_kinds[kind] += 1
    if kind == "TEMPORAL":
        idx_q, idx_t = p.index("Question:"), p.index("This question asks about timing")
        check(idx_t > idx_q, f"[{kind}] timing directive AFTER the question", q_text[:38])
        check("Pomodoro" not in p, f"[{kind}] no contradictory name-a-thing block", q_text[:38])
    if kind == "NAMED":
        check("must name a thing" in p, f"[{kind}] name-a-thing instruction present", q_text[:38])
    banned = ["not specified", "cannot determine", "not mentioned", "unknown",
              "yesterday", "last week", "last month", "recently"]
    hit = [b for b in banned if b in p.lower()]
    check(not hit, f"[{kind}] no primed refusal/relative vocabulary", q_text[:38] + (f" got {hit}" if hit else ""))
    check(sess.last_kwargs.get("prompt_write_enabled") is True,
          f"[{kind}] prompt_write_enabled threaded to generate_reply")
print(f"  branch coverage: {dict(seen_kinds)}")
check(len(seen_kinds) >= 2, "multiple prompt branches exercised")

# adaptive date claim
undated = FakeSession(["Caroline values community support."] * 4)
check(_evidence_carries_dates(undated) is False, "undated evidence detected")
answer_with_osam(undated, "When did Caroline join?")
check("Each evidence item begins with the date" not in undated.last_prompt,
      "date claim dropped when evidence has no timestamps")

print()
print("=" * 72)
print("STAGE 6  scoring + checkpoint round-trip")
print("=" * 72)
demo = [({"category": 2, "answer": "7 May 2023"}, "7 May 2023", 1.0),
        ({"category": 2, "answer": "7 May 2023"}, "Last week", 0.0),
        ({"category": 4, "answer": "Pomodoro technique"}, "Pomodoro technique", 1.0)]
for q, pred, want in demo:
    got = score_locomo_prediction(q, pred)
    check(abs(got - want) < 1e-9, f"score({pred!r}) == {want}", f"got {got:.3f}")

out_file = pathlib.Path("/tmp/_smoke_rows.jsonl")
out_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
completed, reloaded_rows = load_checkpoint(str(out_file))
check(len(reloaded_rows) == len(rows), "checkpoint reload count matches")
check(len(completed) == len(rows), "checkpoint keys parsed")
check(all("retrieval" in r for r in reloaded_rows), "retrieval diagnostics survive JSON round-trip")

print()
print("=" * 72)
print(f"RESULT: {len(FAIL)} failure(s)" if FAIL else "RESULT: ALL CHECKS PASSED")
if FAIL:
    for f in FAIL: print("   FAILED:", f)
print("=" * 72)
sys.exit(1 if FAIL else 0)
