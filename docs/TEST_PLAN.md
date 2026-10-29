# Test plan — IterRet paper revision + δ-mem questions

Created 2026-10-20. Motivated by the two reviews of the IterRet paper (both ask
for component ablations, more benchmarks / another model family, implementation
detail + code; reviewer 2 asks about experience-bank cost, conflicts and scale)
and by the open δ-mem questions. GPU times are rough, for one 24 GB RTX 4090.

Terminology: the paper's Anchor–Facet–Content (AFC) graph is the code's
cue–tag–content (CTC) graph (anchor = cue, facet = tag). "LoCoMo-4" = the first
4 conversations (584 questions); "LME" = MemoryAgentBench LongMemEval
(`longmemeval_s*`, 5 histories × 60 questions, ~363k tokens each).

Status legend: ✅ done · ⏳ queued/running · 🔧 code ready, needs a run ·
⛔ blocked · ✏️ needs code

## Stage 0 — running or blocked

| # | Test | Why | Status | How |
|---|---|---|---|---|
| 0.1 | EventQA 65k (500 q): base vs δ-mem, authors' harness | δ-mem claim where the answer IS in the 64k window | ✅ δ-mem 82.6 vs base 80.4 (+2.2, CI −0.2..+4.6; F1 +2.0, CI +0.5..+3.7) | `MAB_SOURCE=eventqa_65536 bash scripts/run_mab_official.sh` |
| 0.2 | Fact consolidation 32k (sh, mh): base vs δ-mem | second category of the paper (selective forgetting), fits fully | ✅ sh 51 vs 49 (+2, n.s.); mh 4 vs 4 | `MAB_SOURCE=factconsolidation_sh_32k …` |
| 0.3 | δ-mem with the original token-mask logic, LME 1 history | confirm our mask fix is inert in the authors' setup (FlashAttention passes no mask) | ✅ mask never reached (attention_mask=None in all 2,592 calls); legacy == fixed re-run, 60/60 identical | `DELTAMEM_LEGACY_TOKEN_MASK=1 MAB_MAX_TEST_SAMPLES=1 ARMS=deltamem bash scripts/run_mab_official.sh` |
| 0.4 | Official LLM-judge re-scoring of LME base vs δ-mem (300 q) | LME's official metric is an LLM judge, not substring match | ⛔ disk space for the judge model | `DOWNLOAD_JUDGE=1 bash scripts/judge_mab_longmemeval.sh` |

## Stage A — settle IterRet's final configuration (before any ablation)

| # | Test | Why | Status | How |
|---|---|---|---|---|
| A1 | Passages-only + chronological evidence on LoCoMo | +0.107 on Qasper; LoCoMo still uses relevance order + LLM facts | ✅ neutral on LoCoMo-4 (584 q paired): F1 0.4711 → 0.4704 (−0.001, CI −0.020..+0.019), judge 0.711 → 0.731 (+0.021, CI −0.010..+0.053); multi-hop +0.044 F1 (n.s.). Same retrieval cost (19.9 LLM calls/q). `cached_results/stageA/` | `WORKMEM_EVIDENCE_LAYERS=episodic WORKMEM_EVIDENCE_ORDER=document WORKMEM_MAX_SAMPLES=4 WORKMEM_OUTPUT_FILE=… bash scripts/run_pipeline.sh` |
| A2 | IterRet evidence + recent window (LME) | retrieval lost knowledge-update/preference q's (need the latest state) to the window | ✅ no gain at any size (1 history, 20 q, paired, judge): evidence only 0.50; +512 tok 0.50 (identical verdicts); +1024 0.45; +2048 0.50; +8192 0.50. Knowledge-update 1/3 throughout. Likely cause: questions are dated 2023/03–11 inside a multi-year history, so the history's last tokens are not the state at question time. Date-aligned follow-up (window ends at the question's date, `MAB_RECENT_ALIGN=question_date`; Brev A100, `*_A2brev*`): evidence only 0.55; +512 0.50; +2048 0.50; +8192 0.60 (+0.05, CI −0.15..+0.25, 3 wins / 2 losses; knowledge-update 1/3 → 2/3, temporal 4/5 → 3/5). Noise-level at n=20; same-code evidence-only differs 0.50 vs 0.55 across machines, so compare within a machine only. Keep evidence-only; the 8192 date-aligned window is the only candidate worth testing at scale (B2). `cached_results/stageA/` |

## Stage B — headline numbers (final configuration)

| # | Test | Why | Status |
|---|---|---|---|
| B1 | Full LoCoMo (1,540 q) | headline must reflect dates-in-memory + final evidence format | 🔧 after A1 (~30–35 h) |
| B2 | Full LME (300 q): retrieval vs recent window, judge | strongest generalisation evidence (histories ≫ window); only 20 q so far | ✅ 300 q (5 histories), Brev A100, paired: evidence only judge 0.480 / F1 0.381 / subEM 0.313; +8192-tok window ending at question date judge 0.497 / F1 0.391 / subEM 0.327 (judge +0.017, CI −0.027..+0.060, 24 vs 19 wins; F1 +0.011, CI −0.017..+0.039). Gains where the latest state matters (preference +0.133, CI 0..+0.30; knowledge-update +0.067), small losses on single-session-user (−0.044). Not significant overall; graph build ~21 min/history on A100. `cached_results/stageB/` |
| B3 | Qasper, all 200 papers | document QA; 50 papers give ±0.07 CIs | ✅ 200 papers, final config (passages only, paper order; released adapter), Brev A100: F1 0.450, 0 skipped, 11.8 evidence passages/q. `cached_results/stageB/` |

## Stage C — component ablations (main reviewer request)

Each changes ONE thing from the final configuration; run on LoCoMo-4 and LME.
File names get a suffix automatically in `run_mab.sh`; for LoCoMo name
`WORKMEM_OUTPUT_FILE` per ablation.

| # | Ablation | Why | Status | Switch |
|---|---|---|---|---|
| C1 | 1 retrieval round vs 5 | the paper's core claim (iteration helps), never isolated | 🔧 | `ITERRET_MAX_ITERATIONS=1` |
| C2 | AFC graph vs plain embedding top-k (same encoder, same passages, matched size) | does the graph beat ordinary vector search? | 🔧 | `RETRIEVER=flat FLAT_TOP_K=<IterRet's median evidence count>` |
| C3 | experience bank: bootstrapped (held-out) vs empty | R1: contribution; R2: cost/conflicts. A 1-conversation bank previously hurt; all published runs use an empty bank | 🔧 bank from LoCoMo conversations 5–10 (`build_heldout_bank.py`), used on LoCoMo-4 and LME | `bash scripts/run_c3.sh` (`ITERRET_BANK_PATH=…`) |
| C4 | relative-date resolution on/off | R1 praises temporal results; much of that comes from this step | 🔧 **LoCoMo only** (LME passages carry dates as text, so the switch is a no-op there) | `ITERRET_RESOLVE_DATES=0` |
| C5 | facet ranking keyword-only vs keyword+meaning | +0.081 measured on an older config; re-measure | 🔧 | `DISABLE_TAG_EMBEDDER_FUSION=1` |
| C6 | evidence presentation | free from A1 (LoCoMo) / done on Qasper | ✅ Qasper | — |

## Stage D — generalisation and reproducibility

| # | Test | Why | Status |
|---|---|---|---|
| D1 | Second model family (Llama-3.1-8B / Mistral-7B) for graph, search and answering | R1: another LLM family; R2: justify model choice | 🔧 `WORKMEM_ANSWER_BACKEND=vllm` (served model answers, no δ-mem) + Qwen-without-δ-mem reference arm; `D1_MODEL=<hf repo> bash scripts/run_d1.sh` (Llama/Mistral are gated: `hf auth login` first) |
| D2 | Same configuration twice (LoCoMo-4) | R2 doubts reproducibility; quantify run-to-run noise (vLLM nondeterminism) next to released cached graphs | 🔧 |
| D3 | Cost: LLM calls / tokens / seconds per question; graph-build cost | R2: cost and scaling | 🔧 recorded in every new run; `python3 scripts/cost_report.py --results … --graphs …` |

## Stage E — remaining δ-mem questions (lower priority)

| # | Test | Why | Status |
|---|---|---|---|
| E1 | Retrained adapter (w1024) vs base model, Qasper 200 | beat the released adapter by +0.05, never compared with plain Qwen | 🔧 |
| E2 | S memory span with per-passage (message_mean) writes | span (~5 writes) measured only with token writes | 🔧 `TAG=spancheck MAB_MAX_QUESTIONS=1 ARMS=truncated_fullS bash scripts/run_mab.sh` |

## Order

1. Stage 0 (running) → 0.4 when disk is freed.
2. A1, A2 → B1–B3 → C1, C2, C4, C5 (+ C3 once built) → D1, D2 → E1, E2.

Total ≈ 150 GPU-hours. Minimum set answering both reviews: A1, B1, B2, C1, C2, C3, C4, D1.

## δ-mem reproducibility note (found 2026-10-20)

The authors' harness never resets δ-mem's state between questions
(`reset_delta_state=False`). A run restarted mid-way agreed with an
uninterrupted run on only 6/50 answers (same accuracy), while two uninterrupted
runs agreed 60/60: δ-mem's answers depend on which questions were processed
before. Pending (not run, by request): base-model re-run and a reset-per-question
δ-mem run to confirm.

## Results so far that feed the revision

- Tag-layer meaning+keyword fusion: +0.081 F1 full LoCoMo (other team, run 12a→12b).
- Date resolution in memory: temporal +0.283 (paired 300 q).
- Evidence presentation (Qasper 50): 0.355 → 0.462 (= whole paper 0.458).
- LME (20 q): retrieval doubles judge accuracy vs recent window (0.25 → 0.50).
- Experience bank: a 1-conversation bank hurt LoCoMo; every published number uses an empty bank.
- δ-mem vs base, authors' harness, LME 300 q, 64k window: no accuracy difference
  (substring 9.67 vs 10.00, CI −2.33..+1.67); δ-mem refuses far less (66 vs 194).
