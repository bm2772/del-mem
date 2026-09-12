# del-mem fork — Handoff

**Repo:** `github.com/bm2772/del-mem` (branch pushed to `main`).
**Last updated:** 2026-09-12.
**Scope:** this is the handoff for the *bm2772/del-mem fork* of the workmem-vertical
project. It records what THIS fork changed and measured. It does **not** replace
`docs/HANDOFF.md`, which is the parallel session's handoff (vendored at clone
time); read that for the base architecture, the δ-mem math, and the run 1–12
history. This file only covers the fork's own work and current state.

---

## 1. What this fork is, and how it relates to the parallel session

- Forked from `nitinvetcha/C-AIMMS` `workmem-vertical` **before** that session's
  run 12a/12b landed. So the fork started as roughly their **run-11-era** tree.
- The parallel session (resiliente `ashwinkm@…`, see `docs/HANDOFF.md`) then
  produced run 12a (causal-mask fix) and run 12b (tag-layer RRF fusion), reaching
  **0.4214** overall on full LoCoMo.
- This fork independently did the tag-fusion work, then **ported their two
  run-12 fixes** and added retrieval + robustness work on top. Current fork state
  is described below.

**Benchmark:** LoCoMo, `data/locomo10.json` — 10 conversations, 1986 QA, **1540
scored** (category 5 / adversarial excluded). Metric: Porter-stemmed **token-F1**.
The "~584" set some tests use is **the first 4 conversations** (`WORKMEM_MAX_SAMPLES=4`),
not a different dataset.

---

## 2. Current result

**Full LoCoMo (1540 Q), guided path, NO prompt engineering, commit 7239a23:**

| category | fork | parallel run 12b | Δ |
|---|---|---|---|
| **Overall** | **0.4142** | 0.4214 | −0.007 |
| MULTI_HOP | **0.4481** | 0.3614 | **+0.087** |
| SINGLE_HOP | **0.5070** | 0.4766 | **+0.030** |
| TEMPORAL | 0.2223 | **0.4035** | **−0.181** |
| OPEN_DOMAIN | 0.1433 | 0.1732 | −0.030 |

Read: essentially tied overall; the fork's retrieval is **stronger on the
reasoning categories** (multi/single), and the whole deficit is TEMPORAL, which
is a **prompt-format** gap (see §5), not a retrieval/memory gap. The parallel
number almost certainly includes date-grounding prompt formatting; the fork's
0.4142 is on the standard LoCoMo prompt (`OSAM_PROMPT_ENGINEERING=0`).

Validity of that run was confirmed: rounds/q 4.98, routing majority `explicit`
(not fail-open), δ_o live on all 1540, evidence 46/q.

---

## 3. Changes made in this fork (newest first)

- **7239a23 — EM-LLM-style content-graph retrieval (flag-gated, default OFF).**
  Node→node expansion over content nodes, complementing query→node seeding.
  `ctc_graph.py`: `content_similarity_neighbors` (k-NN by embedding cosine,
  cached) and `content_contiguity_neighbors` (timeline-adjacent episodic nodes,
  EM-LLM contiguity buffer). `nodes.py`: expands from the round's cue-gated hits
  before the top-up. NO episode creation. Flags in §4.
- **0d51f27 — vLLM safety guards.** A stale server on port 8000 once made a full
  run silently invalid (every question `fail_open_parse_failed`, n_ev=6, 0.2948).
  `run_pipeline.sh` now waits for the port to be free and verifies the served
  model is ours; the eval early-aborts if the first 15 answered questions are
  ≥90% fully fail-open. See §7.
- **af341c1 — `FALLBACK_TOPUP_ADD` default 5→10.** Conv-0 sweep sweet spot
  (best-or-near-best every category; see §5).
- **4121687 — ported the parallel session's run-12 fixes + down-weighted top-up.**
  (a) causal-mask fix in `delta_impl.py` `_token_validity_mask`; (b) zero-lexical
  collapse in tag-fusion; (c) `FALLBACK_TOPUP_ADD` cap so the top-up supplements
  rather than floods. See §5/§6.
- **0209112 — IterRet retrieval recall.** Semantic-aware tag ranking (RRF fusion,
  the fork's own version of run 12b) + the top-up fallback (fire on thin rounds,
  union not replace).
- **155d579 — `OSAM_DELTA_GAIN` test-time knob + gain-sweep script** (earlier
  δ-mem experiment; see §6).

---

## 4. Flags / knobs (all env, all default to unchanged behaviour)

Retrieval (`IterRet/iterret/nodes.py`):
| var | default | effect |
|---|---|---|
| `ITERRET_FALLBACK_TOPUP_MIN` | 8 | fire the semantic top-up when a round's cue-gated hits < this |
| `ITERRET_FALLBACK_TOPUP_ADD` | 10 | max nodes the top-up adds per round (its weightage) |
| `ITERRET_CONTIGUITY` | 0 | 1 = add timeline-adjacent episodic nodes (EM-LLM contiguity) |
| `ITERRET_CONTIGUITY_WINDOW` | 1 | ± window for contiguity |
| `ITERRET_CONTENT_KNN` | 0 | 1 = add k-NN similarity neighbours of found nodes |
| `ITERRET_CONTENT_KNN_K` | 5 | neighbours pulled per seed node |
| `ITERRET_CONTENT_KNN_ADD` | 10 | max k-NN nodes added per round (query-RRF ranked) |
| `DISABLE_TAG_EMBEDDER_FUSION` | unset | reproduce pure-lexical tag ranking |
| `DISABLE_CONTENT_EMBEDDER_FUSION` | unset | reproduce pure-lexical content ranking |

δ-mem / answering (`deltamem/workmem/osam_workmem.py`, `core/delta_impl.py`):
| var | default | effect |
|---|---|---|
| `OSAM_EVIDENCE_IN_PROMPT` | 1 | 0 = evidence only through S (no-context arm; collapses, see §6) |
| `OSAM_PROMPT_ENGINEERING` | 0 | 1 = add format instructions (temporal/first-person/…); see §5 |
| `OSAM_DELTA_GAIN` | 1.0 | test-time scale on δ-mem's q/o correction |
| `OSAM_PHASE1_GRANULARITY` | message_mean | write granularity (measured not to matter) |
| `WORKMEM_GRAPH_LLM_CHECK_AFTER` | 15 | early-abort window for the dead-vLLM guard |

Eval / run:
| var | effect |
|---|---|
| `WORKMEM_MAX_SAMPLES=N` | process first N conversations. Via `run_pipeline.sh` this now writes a separate `workmem_iterret_n<N>.jsonl` checkpoint (e.g. =4 → the 584-Q set) |
| `WORKMEM_OUTPUT_FILE` | eval output path (run_pipeline sets this itself) |
| `VLLM_PORT` | override if 8000 is taken |

Diagnostics now written per row (`retrieval.*`): `fallback_topup_total`,
`knn_expand_total`, `contiguity_expand_total`, `route_modes`, `stop_reason`,
`evidence_ids`. `osam_contribution.mean_delta_o_ratio` per row for δ-mem.

---

## 5. Key findings

- **The causal-mask bug had δ-mem switched off on the headline path.** Before the
  port, the evidence-in-prompt (incremental) path read `delta_o_ratio = 0.0` on
  every row — the correction was mathematically inert. Fixed; now ~0.034 live on
  all rows. **But making it live did not raise the score** (consistent with the
  parallel session's `delta_scaling` ablation): δ-mem's guidance is ~neutral on
  LoCoMo. The score is driven by retrieval + in-context evidence, not the adapter.
- **~59% of zero-F1 answers are format, not memory failures.** On the conv-0
  guided run, of 46 zeros: 18 temporal relative-vs-absolute date ("yesterday" vs
  "5 July 2023"), 9 first-person leaks ("my daughter" vs "Melanie's daughter"),
  plus paraphrase/number-word cases. Only ~8–10 are genuine retrieval/reasoning
  misses. The retrieval stack is better than token-F1 shows.
- **The TEMPORAL deficit vs the parallel session is that format gap.** Their
  default prompt includes date-grounding; the fork's headline run does not.
  `OSAM_PROMPT_ENGINEERING=1` targets exactly these buckets. Whether to use it is
  a research-integrity call: the first-person/date-grounding parts are legitimate
  task-format alignment; the anti-abstention + LoCoMo-shaped-example parts are
  benchmark-fitting. Recommendation: keep 0.4142 (standard prompt) as the
  architecture number; report `=1` as a labelled ablation only.
- **Top-up weightage: `ADD=10` is the sweet spot.** Conv-0 sweep {5,10,15} scored
  {0.327, 0.371, 0.367}. 5 starves multi-hop; 15 drowns open-domain; 10 is
  best-or-near-best in every category at baseline-level evidence volume (~37/q).
- **`FAIL_OPEN_FALLBACK_TOP_K = 6`** is the tell for a broken graph LLM: a dead
  vLLM makes every round fail-open, so n_ev pins to exactly 6.

---

## 6. δ-mem status (for anyone re-opening the adapter)

- Adapter is the **stock declare-lab δ-mem TSW adapter, trained on Qasper**
  (long-doc QA), rank 8, delta heads q/o, `online_gain=0.05`. It is applied
  **out-of-domain** to LoCoMo dialogue.
- **S-only (`OSAM_EVIDENCE_IN_PROMPT=0`) collapses** (~0.09 on the 584 set; the
  parallel session measured −0.2585 F1 at n=23). r×r = 64 scalars can't carry the
  retrieved evidence; ~12 writes already exceed the 8 independent directions.
- **Gain sweep (`OSAM_DELTA_GAIN` 1/2/4/10)** on the S-only path: `delta_o_ratio`
  scales linearly but F1 does not recover, and at 10× generation degenerates
  ("assistantassistant…"). Test-time scaling cannot make S-only work.
- Open, un-run ideas: test in-domain on **Qasper/LongBench** (the adapter's native
  setting) to separate domain-mismatch from mechanism; retrain with
  `--episode-recent-messages 0` (multi-hour GPU job, needs the training data).

---

## 7. Running it

From the repo root, `source env.sh` first. Outputs live in `$CAIMMS_OUTPUT_DIR`
(= `<workspace>/outputs`, outside the repo).

**Always free the port before a run** (the guards now enforce this, but do it
anyway to be safe):
```bash
pkill -f 'vllm.entrypoints.openai.api_server'; sleep 3; lsof -ti:8000 | xargs -r kill -9
```

- **Smoke (1 conv / 152 Q, ~1h):** `bash scripts/run_pipeline.sh --smoke` → `smoke_results.jsonl`
- **Subset 584 (4 conv), guided only:** `WORKMEM_MAX_SAMPLES=4 bash scripts/run_pipeline.sh` → `workmem_iterret_n4.jsonl`
- **Full (10 conv / 1540 Q, guided only):** `bash scripts/run_pipeline.sh` → `workmem_iterret_full.jsonl`
- **A/B (guided vs S-only), 4 conv by default:** `N=4 bash scripts/ab_evidence.sh`
- **Score:** `python3 scripts/score_calculator.py <file>`
- **Tag/retrieval unit tests (no GPU):** `cd IterRet && python3 -m iterret.tests.test_relevance_ranking`

**Traps (fork-specific; see also `docs/HANDOFF.md` §8):**
1. **Full/subset runs do NOT clear their checkpoint** — archive the target
   `.jsonl` before re-running or you get old rows back on resume.
2. **Keep `outputs/graph_cache/`** — the CTC graphs are unaffected by retrieval/
   ranking/expansion changes; rebuilding costs ~hundreds of vLLM calls per conv.
3. **A stale vLLM invalidates a run silently** — now guarded, but if you see
   uniform `n_ev=6` / `route_modes` all `fail_open_parse_failed`, the graph LLM
   is dead: kill it and restart.

---

## 8. Open / next steps

1. **Prompt-engineering ablation:** `OSAM_PROMPT_ENGINEERING=1` full run — expect
   TEMPORAL ~0.22→~0.40 and overall likely > 0.4214. Report as labelled ablation,
   keep 0.4142 as the architecture number.
2. **EM-LLM channels (7239a23) — measure on the 584 set:** baseline vs
   `ITERRET_CONTIGUITY=1` vs `ITERRET_CONTENT_KNN=1` vs both. Prior: contiguity is
   the safer, more dialogue-appropriate bet (adjacent turns complete answers, low
   drift); k-NN helps multi-hop but risks open-domain. Watch `open_domain` and the
   `knn_expand_total` / `contiguity_expand_total` diagnostics.
3. **LLM-judge secondary metric:** `deltamem/.../llm_judge.py` exists but the eval
   scores token-F1 only. A judge would credit "twice"="2", "Yeah"="Yes",
   "my daughter"="Melanie's daughter" — separating format artifacts from real
   misses without any prompt-fitting. Cleanest honest way to show retrieval quality.
4. **In-domain δ-mem test on Qasper/LongBench** (see §6).
