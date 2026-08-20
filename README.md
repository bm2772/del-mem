# C-AIMMS — WORKMEM vertical

Integration of **δ-mem** (an online associative-memory adapter for a frozen LLM) with
**IterRet** (iterative Cue-Tag-Content graph retrieval), evaluated on **LoCoMo**.

IterRet retrieves evidence for a question → that evidence is written into δ-mem's
online state `S` (OSAM) → the frozen Qwen3-4B backbone generates an answer with `S`
steering its attention via low-rank corrections.

## Start here

**[`docs/HANDOFF.md`](docs/HANDOFF.md)** — the single source of truth: current results,
open issues, measured findings that shouldn't be re-derived, and the traps that will
otherwise cost you a 30-hour run. Read it before changing anything.

## Layout

```
docs/
  HANDOFF.md                  project state, results, open issues, traps  <- READ FIRST
  ENVIRONMENT.md              environment + install notes
  C-AIMMS_Pipeline_Teardown.pdf
env.sh                        all paths, every machine -- source this first
IterRet/                      retrieval vertical (vendored, the ONLY copy)
  iterret/                    ctc_graph.py, nodes.py, memory_builder.py, ...
delta-Mem/
  deltamem/
    core/delta_impl.py        OSAM write/read math (DeltaMemAttention.forward)
    runtime/session.py        DeltaMemChatSession: generate_reply, _ingest_full_ids
    eval/locomo_protocol.py   LoCoMo scoring + shared category constants
    workmem/
      eval_locomo_iterret_mock.py   MAIN EVAL LOOP
      osam_workmem.py               Phase 1/2 OSAM population + all prompts
      iterret_bridge.py             wraps IterRet's retrieve/reflect loop
      ab_write_granularity.py       paired A/B harness
  data/locomo10.json          the benchmark
scripts/                      run_pipeline.sh, setup_env.sh, download_assets.sh,
                              dryrun_pipeline.py, score_calculator.py, slurm + guardian
trained_models/               adapter config (weights not in git)
requirements_exact.txt        authoritative pinned environment
```

## Running

```bash
source env.sh
bash scripts/run_pipeline.sh --smoke     # 152 questions, ~1h
bash scripts/run_pipeline.sh             # all 1540, ~30-35h
```

`env.sh` sets `PYTHONPATH` and every path from its own location, so the same tree runs
unchanged on the Mac, Mahamathi and resiliente-2003. It expects `models/` and `outputs/`
in the repo's **parent** directory (override with `CAIMMS_WORKSPACE`).

Full instructions, including the checkpoint trap that silently returns stale scores,
are in [`docs/HANDOFF.md`](docs/HANDOFF.md) section 3.

## Note on IterRet

`IterRet/` is **vendored into this branch** and is the only copy — a clone of
`workmem-vertical` is self-contained and gets the retrieval fixes the current results
depend on (IDF/RRF content ranking, fail-open fallback via the fused ranker, routing
token budget, retrieval diagnostics).

Upstream IterRet lives on the `IterRet` branch of this same repo and is owned by Pragnya;
these modifications have **not** been reconciled with her copy. The exact diff is preserved
at `docs/patches/iterret-workmem-modifications.patch` for that reconciliation. See
`docs/HANDOFF.md` section 9.
