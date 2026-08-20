# IterRet modifications, preserved for reconciliation

`iterret-workmem-modifications.patch` is the exact diff between Pragnya's upstream
`IterRet` branch (commit `949d4a3`) and the version vendored into this branch at
`workmem-vertical/IterRet/`.

It exists because the standalone `IterRet/` checkout was removed on 2026-08-20 when
the two copies were unified (see docs/HANDOFF.md section 9). The vendored code is the
canonical copy; this patch is what you hand Pragnya, or replay onto a fresh clone of
the `IterRet` branch, when reconciling.

## What's in it

| file | change |
|---|---|
| `iterret/ctc_graph.py` | IDF-weighted, stopword-stripped relevance ranking + RRF fusion with embeddings |
| `iterret/nodes.py` | fail-open fallback uses the fused ranker (not cosine alone); `ROUTING_MAX_TOKENS=512`; per-round route diagnostics |
| `iterret/llm_client.py` | per-call `max_tokens` override on `chat()` |
| `iterret/state.py` | diagnostic fields on `IterRetState` |
| `iterret/locomo_data.py` | minor local tuning |

Two files are **new**, so they are not in the patch (a diff has nothing to diff against) —
take them from `workmem-vertical/IterRet/iterret/`:
`episode_segmenter.py` and `tests/`.

## Replaying it

    git clone -b IterRet https://github.com/nitinvetcha/C-AIMMS.git iterret-upstream
    cd iterret-upstream
    git apply /path/to/iterret-workmem-modifications.patch
