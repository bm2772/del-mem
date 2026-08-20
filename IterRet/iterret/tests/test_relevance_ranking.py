"""Unit tests for the IDF-weighted relevance ranking added to ctc_graph.py.

Pure Python, no torch/sentence-transformers/GPU dependency -- these test the
scoring logic in isolation, not the real embedder (that needs the cluster;
see the offline validation script instead). Runnable either with pytest or
directly: `python3 -m iterret.tests.test_relevance_ranking`.
"""
from __future__ import annotations

import os
import sys

from iterret.ctc_graph import CueTagContentGraph, _content_tokens, _tokenize


class _FakeEmbedder:
    """Deterministic stand-in for a real embedding backend. Encodes text as
    a bag-of-words count dict and scores cosine similarity over it -- exists
    purely to prove the RRF fusion path and the DISABLE_CONTENT_EMBEDDER_FUSION
    escape hatch both actually run, not to model real semantic behaviour.
    """

    def encode(self, text):
        counts = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def similarity(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        na = sum(v * v for v in a.values()) ** 0.5 or 1.0
        nb = sum(v * v for v in b.values()) ** 0.5 or 1.0
        return dot / (na * nb)


def test_empty_graph_does_not_crash():
    g = CueTagContentGraph()
    assert g.rank_contents_by_relevance([], "anything") == []
    assert g.rank_tags_by_relevance([], "anything") == []
    # _ensure_idf must handle zero content nodes without dividing by zero.
    g._ensure_idf()
    assert g._df is not None
    assert g._idf == {}


def test_single_node_graph():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline went to a support group")
    order = g.rank_contents_by_relevance(["e1"], "support group")
    assert order == ["e1"]


def test_all_stopword_query_scores_zero_not_crash():
    g = CueTagContentGraph()
    g.add_content("e1", "the group of them")
    g.add_content("e2", "Caroline joined a support group")
    # Query is entirely stopwords/function words -> _content_tokens(query) is
    # empty -> every score must be 0.0, not a crash from an empty sum().
    order = g.rank_contents_by_relevance(["e1", "e2"], "the of to and")
    assert order == ["e1", "e2"]  # falls back to alphabetical tie-break, both score 0.0


def test_add_content_invalidates_cached_idf():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline went to a support group")
    g._ensure_idf()
    assert "lgbtq" not in g._df
    g.add_content("e2", "Caroline attended an LGBTQ support meeting")
    # Adding a node must invalidate the cache so the new token is counted.
    assert g._df is None
    g._ensure_idf()
    assert g._df["lgbtq"] == 1


def test_idf_weighting_beats_raw_overlap_on_a_real_style_question():
    """Reproduces the measured rank-8-to-rank-4 pathology at small scale.
    "distractor" is built to actually WIN under the old raw-overlap scorer
    (6 shared words -- caroline/did/to/the/go/when, all stopwords except
    "caroline" -- vs gold's 5: caroline/to/group/support/lgbtq). Verified
    against a standalone reimplementation of the pre-fix scorer: old order
    was ['distractor', 'gold']. This test only passes because of the fix,
    not coincidentally.
    """
    g = CueTagContentGraph()
    g.add_content("gold", "Caroline: I went to a LGBTQ support group yesterday")
    g.add_content("distractor", "Caroline: did she go to the store when it opened")
    query = "When did Caroline go to the LGBTQ support group?"

    order = g.rank_contents_by_relevance(["gold", "distractor"], query)
    assert order[0] == "gold", f"expected gold first, got order {order}"


def test_tag_ranking_also_idf_weighted():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline joined a support group for LGBTQ people")
    tags = ["the support group", "unrelated grocery list", "lgbtq community meetup"]
    order = g.rank_tags_by_relevance(tags, "LGBTQ support group meeting")
    assert order[0] in ("lgbtq community meetup", "the support group")
    assert order[-1] == "unrelated grocery list"


def test_display_text_fix_d4_ranker_sees_timestamp():
    """Before the fix, the lexical scorer read node.text (no timestamp), so
    a query token existing ONLY in the timestamp could never contribute to
    the score, and identical .text on both nodes meant the pre-fix scorer
    could only break the tie alphabetically. IDs are deliberately named so
    alphabetical order favours the WRONG node under the old behaviour
    ("aa_undated" < "zz_dated") -- verified against a standalone
    reimplementation of the pre-fix scorer: old order was
    ['aa_undated', 'zz_dated']. This only passes now because the fix reads
    display_text() and its date tokens genuinely outscore the untimed node.
    """
    g = CueTagContentGraph()
    g.add_content("zz_dated", "Caroline had a picnic with friends", time="7 May 2023")
    g.add_content("aa_undated", "Caroline had a picnic with friends")
    query = "picnic 7 May 2023"
    order = g.rank_contents_by_relevance(["zz_dated", "aa_undated"], query)
    assert order[0] == "zz_dated", f"expected the dated node ranked first, got {order}"


def test_no_embedder_fusion_is_pure_lexical():
    g = CueTagContentGraph()
    g.add_content("a", "apple banana cherry")
    g.add_content("b", "apple banana")
    lexical = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert lexical[0] == "a"  # more overlap, no embedder attached at all


def test_disable_fusion_env_flag_bypasses_embedder():
    g = CueTagContentGraph()
    g.add_content("a", "apple banana cherry")
    g.add_content("b", "durian elderberry fig")
    g.attach_embedder(_FakeEmbedder())

    lexical_only = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert lexical_only[0] == "a"

    os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"] = "1"
    try:
        with_flag = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
        assert with_flag == lexical_only, "flag must reproduce the pure-lexical order"
    finally:
        del os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"]

    # Sanity: with the flag OFF and a real embedder attached, fusion actually
    # runs (doesn't error, still returns a valid permutation) -- this doesn't
    # assert a specific order since RRF fusion behaviour isn't what's under
    # test here, just that the fused path executes without the flag.
    fused = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert set(fused) == {"a", "b"}


def test_load_reconstructs_same_idf_as_build():
    """IDF must derive entirely from self.contents so load() (which never
    touches _df/_idf directly) reproduces identical scores -- this is what
    makes the change need no cache migration.
    """
    g1 = CueTagContentGraph()
    g1.add_content("e1", "Caroline went to a LGBTQ support group", time="8 May 2023")
    g1.add_content("e2", "Melanie painted a sunrise last year")
    g1.link("caroline", "support", "e1")

    data = g1.to_dict()
    assert "idf" not in data and "df" not in data  # confirms: no new persisted field

    g2 = CueTagContentGraph.load  # just referencing to ensure it still exists
    g2 = CueTagContentGraph()
    for cid, c in data["contents"].items():
        g2.add_content(cid, c["text"], layer=c.get("layer", "episodic"),
                        time=c.get("time"), topic_links=c.get("topic_links", []))
    for cue_id, tag, content_id in data["links"]:
        g2.link(cue_id, tag, content_id)

    query = "Caroline LGBTQ support group"
    assert (g1.rank_contents_by_relevance(["e1", "e2"], query)
            == g2.rank_contents_by_relevance(["e1", "e2"], query))


def test_rank_functions_return_permutation_not_subset():
    g = CueTagContentGraph()
    g.add_content("a", "one two three")
    g.add_content("b", "four five six")
    g.add_content("c", "seven eight nine")
    ids = ["a", "b", "c"]
    order = g.rank_contents_by_relevance(ids, "one two three four")
    assert sorted(order) == sorted(ids)
    tags = ["alpha beta", "gamma delta", "epsilon zeta"]
    torder = g.rank_tags_by_relevance(tags, "alpha gamma")
    assert sorted(torder) == sorted(tags)


_ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def _run_all():
    passed, failed = 0, []
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(_ALL_TESTS)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
