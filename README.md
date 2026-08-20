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
IterRet/                      retrieval vertical (vendored -- see note below)
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
scripts/                      slurm + guardian + scoring helpers
trained_models/               adapter config (weights not in git)
requirements_exact.txt        authoritative pinned environment
```

## Running

Full instructions, including the checkpoint trap that silently returns stale scores,
are in [`docs/HANDOFF.md`](docs/HANDOFF.md) section 3. `PYTHONPATH` must include both
`delta-Mem/` and `IterRet/`.

## Note on IterRet

`IterRet/` is **vendored into this branch** so the vertical is self-contained — a clone
of `workmem-vertical` gets the retrieval fixes the current results depend on (IDF/RRF
content ranking, fail-open fallback via the fused ranker, routing token budget, retrieval
diagnostics). Upstream IterRet lives on the `IterRet` branch of this same repo and is
owned by Pragnya; these modifications have **not** been reconciled with her copy. See
`docs/HANDOFF.md` section 9.
