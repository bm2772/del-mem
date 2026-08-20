# C-AIMMS / WORKMEM — Handoff

**Last updated:** 2026-08-19. This file replaces all prior versions (the run-by-run history that used to live here has been trimmed — only conclusions worth not re-deriving are kept, in §7).

---

## 1. What this is

**The project's own spec is `~/Downloads/COLM_Cognitive_AI_Memory_Architecture_Project (1).pdf`** — read this before touching `osam_workmem.py` or `delta_impl.py` again. It formally defines four modules: HETREP (heterogeneous encoding — not built), ADASTORE (adaptive storage/hierarchy — not built), ITERRET (the retrieval graph — built, this is `IterRet/`), and **WORKMEM** (§1.5 of the paper — the OSAM integration this handoff is about).

The underlying memory mechanism is **δ-mem** (arXiv:2605.12357, also in `~/Downloads/`): a PEFT adapter giving a frozen LLM an online-writable associative memory, one **r×r state matrix per attention layer** (r=8 here), updated by a gated delta rule:
```
S_t = λ_t·S_{t-1} + β_t·(v_t - S_{t-1}k_t)·k_tᵀ      (λ_t = 1 - β_t)
```
Read: `r_t = S_{t-1}·q_t`, projected into `Δq_t, Δo_t` which correct the backbone's query and attention output (`q̃ = q⁰ + (α/r)Δq`, `ȳ = a + (α/r)Δo`). Only q/o heads are active in this adapter (`delta_heads: ["q","o"]`); k/v corrections are zero. **The write is not additive — it erases along the new key's direction before writing.**

**The paper's own architectural principle, stated three separate times** (abstract: "keeping the prompt context uncorrupted"; §1.5 prose: "The naive approach — injecting E_T as additional prompt tokens — consumes token budget, suffers from retrieval noise... Instead, C-AIMMS routes memory influence through δ-mem"; eq. 10 caption: "the prompt context q_T carries no retrieved content"): **retrieved evidence should influence generation only through S, never by also being placed in the prompt.** See §6 item 1 — the current pipeline does not follow this.

**Two-phase OSAM, per the paper (§1.5, "Interaction with ITERRET"):** Phase 1 populates S from the retrieved evidence set E_T, *offline, before generation begins*. Phase 2 writes *incrementally from the model's own newly generated tokens* during response generation. This is a spec about **what gets written when** — not about write granularity (message_mean vs token), which is a separate, orthogonal axis the paper studies (§3.5 of the arXiv paper) as three complete alternative training configurations (TSW/SSW/MSW), not a mid-inference switch.

Benchmark: **LoCoMo**, 1540 scored questions (adversarial excluded — its correct answer is a refusal, which rides the same bias hurting every other category). Four categories: multi-hop, temporal, open-domain, single-hop. Metric: token-level F1, Porter-stemmed.

---

## 2. Where things live

| | |
|---|---|
| **Local Mac** | `~/Documents/C-AIMMS/` — `workmem-vertical/` is the repo (branch `workmem-vertical`, remote `nitinvetcha/C-AIMMS`). `IterRet/` alongside it is the legacy standalone checkout on the `IterRet` branch; **the canonical copy is now vendored at `workmem-vertical/IterRet/`** — see §9. `outputs/` and `models/` are deliberately outside the repo. |
| **Transfer bundle** | `~/Documents/ashwinGPU/` — what runs on resiliente-2003. Four files are `CAIMMS_ROOT`-path-patched; merge changes into these, don't blind-copy. |
| **resiliente-2003** | `ashwinkm@10.24.32.171`, `~/ashwinGPU/`. 2× RTX 4090, no scheduler. **Primary run environment.** |
| **Mahamathi** | `arnavbhatt@mahamathi2`, SLURM, 8× A100. Secondary — see §8 for its unresolved issue. |
| **Papers** | `~/Downloads/COLM_Cognitive_AI_Memory_Architecture_Project (1).pdf` (this project's spec), `~/Downloads/2605.12357v1.pdf` (δ-mem method paper), `~/Downloads/Version_2.pdf`. Not in the repo. |

Key files (repo root = `workmem-vertical/`):
```
docs/HANDOFF.md                 this file
docs/ENVIRONMENT.md             environment notes
scripts/                        slurm + guardian + score_calculator.py
delta-Mem/deltamem/
  core/delta_impl.py            OSAM math (DeltaMemAttention.forward) — verified against paper eq. 1-11, see §6
  runtime/session.py            DeltaMemChatSession: generate_reply, _ingest_full_ids
  eval/locomo_protocol.py       scoring + shared category constants
  workmem/
    eval_locomo_iterret_mock.py MAIN EVAL LOOP
    osam_workmem.py             populate_osam_from_evidence, answer_with_osam, all prompts
    iterret_bridge.py           wraps IterRet's loop, now carries retrieval diagnostics
    ab_write_granularity.py     paired A/B harness (granularity — result in §7)
IterRet/iterret/                VENDORED into this branch (see §9)
  ctc_graph.py, nodes.py        retrieval graph + ranking (out of scope for §5 — that's delta-mem only)
```

---

## 3. How to run

```bash
rsync -av --exclude='.hf' --exclude='outputs' --exclude='__pycache__' ~/Documents/ashwinGPU/ ashwinkm@10.24.32.171:~/ashwinGPU/
ssh ashwinkm@10.24.32.171
cd ~/ashwinGPU/outputs && mv workmem_iterret_full.jsonl "workmem_iterret_full_$(date +%Y%m%d_%H%M%S).jsonl"   # §8 TRAP 1 — full run does NOT clear its own checkpoint
tmux new -s caimms
bash ~/ashwinGPU/run_pipeline.sh              # full, ~30-35h. Add --smoke for 152q/~1h
```

No-GPU static check before committing to a run: `cd ~/ashwinGPU && PYTHONPATH=".:IterRet:delta-Mem" python3 dryrun_pipeline.py`

Score: `python ~/ashwinGPU/score_calculator.py ~/ashwinGPU/outputs/workmem_iterret_full.jsonl`

Pull results back: `rsync -avP ashwinkm@10.24.32.171:~/ashwinGPU/outputs/workmem_iterret_full.jsonl ~/Documents/C-AIMMS/outputs/workmem_runNN_<date>.jsonl` (rename on arrival — never overwrite a prior run's file).

**Env knobs**, all default to current pipeline behavior:

| var | default | effect |
|---|---|---|
| `OSAM_PHASE1_GRANULARITY` | `message_mean` | measured not to matter (§7) |
| `OSAM_PHASE2_PROMPT_WRITE` | `1` | `0` = prefill reads S without overwriting it — **this is the closest existing knob to the paper's actual Phase 2 spec, see §6 item 2** |
| `WORKMEM_TEMPORAL_NARROWING` | `0` | `1` restores temporal evidence narrowing (was hurting the category, see §7) |
| `OSAM_TEMPORAL_QUERY_PATTERN` | built-in regex | override the timing-question detector |

---

## 4. Results

All numbers recomputed directly from the `.jsonl` files, adversarial excluded.

| run | date | overall | multi-hop | temporal | open-dom | single-hop |
|---|---|---|---|---|---|---|
| run 6 | Jul 7 | 0.2756 | 0.2192 | 0.3015 | 0.0960 | 0.3051 |
| run 8 | Aug 12 | 0.2935 | 0.2412 | 0.2697 | 0.2014 | 0.3306 |
| run 9 (+IDF ranking) | Aug 13 | 0.3016 | 0.2582 | 0.2430 | 0.2183 | 0.3479 |
| run 10 (resiliente port) | Aug 16 | 0.2906 | 0.2311 | 0.2343 | 0.2090 | 0.3413 |
| **run 11** (Aug-19 prompt/retrieval fixes) | Aug 19 | **0.3139** | 0.2422 | **0.3233** | **0.1349** | 0.3547 |

**Run 11 is the best overall result.** Temporal broke its four-run losing streak decisively (+0.080 vs run 9, matched-pair: 113 improved / 41 regressed / net +25.78 on 321 questions) — driven by two prompt changes: the timing instruction moved to immediately after the question (restoring run 6's structure), and its enumerated relative-expression list ("yesterday, last week...") removed (was priming the model to emit exactly those phrases). Open-domain regressed sharply (−0.083) — root-caused to the new yes/no instruction ("start with Yes/No, then add a short phrase") combined with `score_locomo_prediction`'s pre-existing category-3 behavior of truncating gold at the first `;`, so any elaboration is precision-poison against a one-word target. Measured: 0/49 yes/no-gated questions elaborated in run 9 across all categories; 23/23 did on open-domain alone in run 11, scoring 0.145 mean F1. A second, smaller new issue: the "name a thing" instruction has no case for count questions (`"How many dogs?"` → `"Scout"`, a dog's name, not a number) — F1 on "how many" questions fell 0.190→0.122. Neither has been fixed yet.

---

## 5. Delta-Mem issues, prioritized (paper-grounded)

Full writeup with code citations and the mechanism given in the session that produced this handoff — summarized here so it isn't re-derived.

1. **Retrieved evidence is placed in the prompt, which the paper explicitly says not to do.** `populate_osam_from_evidence` sets `session.messages` to the evidence list; nothing clears it before `generate_reply` tokenizes `self.messages` for Phase 2, so evidence text is in the backbone's attended context on every question. This directly contradicts the paper's stated principle (three places, quoted in §1).

    **Tested 2026-08-20, then reverted — result is real, keep it in mind.** A toggle was built (write evidence into S in Phase 1 as normal, then wipe the text/KV context before Phase 2 without touching S) and paired A/B'd on resiliente-2003, n=23: removing evidence from the prompt cost **-0.2585 F1** (95% CI **[-0.4307, -0.0830]**, entirely negative, no_evidence worse on 14/23, identical on 0/23). Matches the δ-mem paper's own no-context ablation (46-51%→8.05%) in direction and rough size — **confirms the risk this item warned about was real**, on this adapter, as currently trained. Per explicit instruction, the toggle, its session method, and its A/B harness have all been removed from the codebase — this bullet is what's left of the experiment. Don't re-implement it casually; if revisited, it needs OSAM's standalone signal to be stronger first (see item 4's new sub-finding).
2. **Phase 2 writes the instruction+question block into S, which the paper's Phase 2 does not describe.** The paper's Phase 2 is "incrementally from the model's own newly generated tokens" — i.e. the answer being produced, not a ~250-330 token instruction/question prefill. `OSAM_PHASE2_PROMPT_WRITE=0` makes the prefill read-only, matching the paper's actual spec more closely than the current default. Not yet run. (This knob is unrelated to item 1's reverted one and is still in the code.)

3. **`online_gain=0.05` / `last_delta_o_ratio` — now logged, kept in the code.** Every result row carries `osam_contribution` (mean/max delta_o-vs-base_o ratio per question, via `collect_delta_mem_output_ratio_stats`, snapshotted in `generate_reply` right after the Phase-2 prefill). This is the instrument that produced item 1's measurement above.

    **New bug found while using it — not yet fixed, see §8.** On the default code path (evidence in prompt, i.e. every full run so far), `osam_contribution` reports exactly `0.0` for base_o_norm, delta_o_norm, *and* the ratio — not a small real number, a broken reading. Likely cause (correlated, not yet pinned to one line): that path's Phase-2 prefill is an *incremental* ingest — Phase 1's evidence KV cache is still attached, only the new instruction+question suffix gets a fresh forward pass — and no explicit `attention_mask` is ever passed to the model call in `_ingest_full_ids`. Something about deriving the token-validity mask for that specific pattern produces an all-invalid mask, which the masked-norm helpers correctly report `0` for. The `no_evidence` arm (a *fresh* ingest, no prior cache) measured fine and consistently (~0.023-0.025 ratio) across every question in the same n=23 run, which is what makes item 1's number trustworthy despite this. Not a new bug — same untouched `delta_impl.py` as always — just never observable before nothing had ever called this function.
4. **Two new prompt-side issues from run 11** (see §4): the yes/no instruction needs a category-3-aware exception or should drop "then add a short phrase"; the named-thing instruction needs a fourth branch for `^how\s+many\b` questions that asks for a count, not a name.

---

## 6. Other open issues

- **`match_query_to_cues` hard `issubset` stopword gate** — retrieval-side, out of scope for §5, but still the suspected largest ceiling. Graph caches now exist locally (pulled from resiliente-2003), so it's finally measurable offline.
- **Evidence strings duplicate timestamp/speaker** — `memory_builder` prepends `Speaker:`, `display_text()` prepends `[ts]` again on top of turn text that already has both.
- **Topic layer is dead code** — `_abstract_topics` spends LLM calls building nodes that are never `.link()`'d, so they're unreachable. Deleting the call is a pure speed win.
- **`information_gaps` seeds with a literal sentinel string** that only clears if the LLM echoes it verbatim, likely making the early-exit route dead.
- **`evidence_filter` silently no-ops if MiniLM is unavailable** — `KeywordOverlapEmbeddingBackend.encode` returns a dict, `_cosine` assumes float lists, `zip` + `TypeError` gets swallowed by a bare `except`. `run_pipeline.sh` preflights against this; `eval_locomo_ablation.py` and `ab_write_granularity.py` do not. Not fixed.
- **Nothing is committed** — five+ weeks of work across two repos, uncommitted.

---

## 7. Durable findings — don't re-derive these

- **Granularity genuinely doesn't matter, measured properly.** Paired A/B, resiliente-2003, n=380: `message_mean` vs `token` → +0.0005 F1, 95% CI [−0.0106, +0.0107], 63.4% identical predictions (confirms the arms took genuinely different code paths, not a repeat of an earlier `role="system"` bug that made Phase 1 silently fall back to token-only writes for every run through run 10). See §5 item 3 for why this is unsurprising given the adapter's training.
- **Removing evidence from the prompt costs a lot, measured once.** Paired A/B, resiliente-2003, n=23: -0.2585 F1, 95% CI entirely negative [−0.4307, −0.0830]. See §5 item 1. The mechanism that produced this has been reverted out of the code; this number is what's left and shouldn't need re-deriving unless the adapter itself changes.
- **Abstention is not calibration.** r = −0.03 between abstaining and whether the gold evidence is actually present — the model refuses on ~27% of questions whose evidence it has and attempts ~70% of those it doesn't. Abstaining is never correct on cats 1-4 (0/1540 golds is a refusal) and scores 0.028 mean F1 vs 0.397 for answering.
- **Temporal length-band effect**: gold answers average 3.35 words; only the 3-5 word band can hold a date, and it scores 0.42-0.52 F1 in every run measured. The category's score tracks the population of that band far more than any change in the model's actual date-finding ability.
- **Retrieval scorer comparison** (offline, gold LoCoMo evidence): raw token overlap R@12=0.469; +stopwords+IDF → 0.624; BM25 adds nothing beyond that (default b=0.75 is *worse* than plain IDF on short turns); cosine alone is worst (R@1 0.186). 98.3% of gold turns missed at k=12 still share a content word with the query — it's a ranking problem, not a reachability one.
- **Capacity**: state is r×r=64 scalars per layer; ~12 evidence writes already exceed the 8 independent directions available. Not the only channel though — see §5 item 1, the evidence is also (currently, wrongly per spec) in the KV cache.
- Open-domain is structurally hardest — the paper's own no-OSAM baseline gets 0.1894/10.77 there. Date arithmetic is unreliable at 4B regardless of prompting. At least one LoCoMo gold has a typo (`"Yesteammates..."`) that zeroes a correct answer.

---

## 8. Traps

1. **Full run does not clear its checkpoint** (only `--smoke` does). Archive `workmem_iterret_full.jsonl` before every full run or you'll get old scores back in 2 minutes.
2. **Keep `outputs/graph_cache/`** — unaffected by prompt/ranking changes, costs ~600 vLLM calls/conversation to rebuild.
3. **Generation is greedy** (`do_sample=False`), not the official protocol's `temp=0.4/top_p=0.9/top_k=10`. `max_new_tokens` defaults to 2048, not the protocol's 50 (`OFFICIAL_MAX_NEW_TOKENS` imported, unused) — harmless today since answers are terse, but a footgun.
4. **`hasattr(graph, "nodes")` is always False** — "Graph ready: N nodes" has never printed a real number. Cosmetic.
5. **`osam_contribution` on a full run right now will show `0.0`/`0.0`/`0.0` on every row.** Not a real measurement — see §5 item 4. Don't conclude "OSAM contributes nothing" from it until the incremental-ingest masking bug behind it is found and fixed.

---

## 9. Ownership and admin — unfinished

- **`IterRet/` is now VENDORED into `workmem-vertical`** (2026-08-20). It lives at `workmem-vertical/IterRet/` and is tracked by this branch, so a clone of `workmem-vertical` is self-contained and gets the retrieval fixes the current results depend on (IDF/RRF content ranking, fail-open via the fused ranker, `ROUTING_MAX_TOKENS`, retrieval diagnostics). Previously it was a separate checkout on the `IterRet` branch, meaning anyone cloning this branch alone got a pipeline that couldn't reproduce the results.
  - **It is still Pragnya's vertical and still unreconciled with her copy.** Vendoring made the branch self-contained; it did **not** resolve ownership. The upstream `IterRet` branch is untouched by this — these changes have not been pushed there.
  - **Drift risk, now live:** the legacy standalone checkout at `~/Documents/C-AIMMS/IterRet/` still exists locally and is byte-identical as of vendoring. Two copies of the same code is exactly how this project already produced three divergent `CATEGORY_MAP` definitions. Treat `workmem-vertical/IterRet/` as canonical and retire the standalone one once the cluster `PYTHONPATH`s are repointed.
- **Mahamathi job kills**: `cn6`/`a100` jobs get `CANCELLED by 0` (root) after 40min-1.5h. Ruled out: QOS, preemption, self-cancellation. Leading hypothesis: oversubscribed-node contention. **Admins never asked.** Largely moot while resiliente-2003 is primary.
- **`pydantic` version untested**: `setup.md` says `2.9.2` is critical for vLLM/FastAPI; `requirements_exact.txt` pins `2.13.4`, which is what both boxes actually ran successfully. Which one actually matters has never been isolated.
