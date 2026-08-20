
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

ContentLayer = Literal["episodic", "semantic", "topic"]

_WORD_RE = re.compile(r"[a-z0-9]+")

# Function words carry almost no retrieval signal but score a full point under
# unweighted token overlap, same as a rare entity word -- this is why e.g. a
# turn matching {the, to, when} could previously out-rank one matching
# {lgbtq}. Verified against this project's actual corpus (locomo10.json), not
# assumed: no collisions with any of the 18 speaker names; "may" is
# deliberately absent (see below); an IDF-aware adaptive variant (keep a
# stopword only when ITS OWN corpus IDF is high) was measured across five
# thresholds and was uniformly worse than this flat list (0.579-0.610 vs
# 0.614 recall@12 of gold LoCoMo evidence) -- this is a measured choice, not
# a compromise.
#
# "may" is NOT in this list, unlike most stoplists: it collides with the
# month name (tokenization lowercases "May" -> "may"). Measured impact: 29
# LoCoMo questions reference a "May" date and 26 evidence timestamps contain
# it too, so stripping it costs the query and its own evidence their only
# shared anchor word simultaneously. Removing it took recall@12 on the
# affected questions from 0.5714 to 0.6190 with no measurable change (+0.0007)
# on the rest of the corpus.
#
# re/ve/ll/d/m/won are contraction remnants ("we're", "I've", "we'll", "I'd",
# "I'm", "won't") -- added for correctness; measured to have no effect here
# (their corpus frequency is high enough that IDF already assigns them near-
# zero weight), kept anyway since they cost nothing.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
of in on at to for with from by about as into over after before between
out up down is are was were be been being am do does did doing done
have has had having i you he she it we they me him her them my your his
its our their what when where who whom which why how whose not no nor
so too very can could will would shall should might must s t don
didn isn aren wasn weren said says say also just one two some any all
more most other such own same only re ve ll d m won
""".split())


def _tokenize(text: str) -> Set[str]:
    """Raw token set -- UNCHANGED. match_query_to_cues (the search/gating
    layer) depends on this exact behaviour and is deliberately out of scope
    for this change; keeping this function untouched means there is no
    shared code path that could be broken by the relevance-scoring rewrite
    below.
    """
    return set(_WORD_RE.findall(text.lower()))


def _content_tokens(text: str) -> Set[str]:
    """Token set for RELEVANCE SCORING ONLY (rank_tags_by_relevance /
    rank_contents_by_relevance): stopwords removed. Kept as a separate
    function from _tokenize precisely so the search-layer caller
    (match_query_to_cues) is structurally unaffected by anything below.
    """
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


@dataclass
class CueNode:
    cue_id: str
    tag_set: Set[str] = field(default_factory=set)


@dataclass
class ContentNode:
    content_id: str
    text: str
    layer: ContentLayer = "episodic"
    time: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    topic_links: List[str] = field(default_factory=list)  # for layer == "topic": episode ids

    def display_text(self) -> str:
        """Text as it should be shown to the LLM: episodic memories carry a
        timestamp (MRAgent Sec. 3.2: "episodic memories are further
        organized along a unified timeline, allowing temporal constraints
        to be imposed during reconstruction") -- without surfacing it here,
        that timeline is stored but never actually reaches the model, so
        temporal questions become unanswerable even when the right episode
        was retrieved.
        """
        return f"[{self.time}] {self.text}" if self.time else self.text


class CueTagContentGraph:
    """The CTC memory graph plus its traversal operators."""

    def __init__(self) -> None:
        self.cues: Dict[str, CueNode] = {}
        self.contents: Dict[str, ContentNode] = {}
        # phi_(c,g)->v : (cue, tag) -> content ids
        self.cue_tag_to_content: Dict[Tuple[str, str], Set[str]] = {}
        # phi_v->(c,g) : content id -> {(cue, tag)}
        self.content_to_cue_tag: Dict[str, Set[Tuple[str, str]]] = {}
        # Optional semantic matching (attach_embedder). When absent, all matching
        # stays lexical (token-overlap) -- so the dialogue/LoCoMo path is unchanged.
        self._embedder = None
        self._content_emb: Dict[str, object] = {}
        self._cue_emb: Dict[str, object] = {}
        # Corpus statistics for relevance scoring (IDF over content nodes).
        # Built lazily on first rank_*_by_relevance call, invalidated by
        # add_content. Derived entirely from self.contents, so load()
        # reconstructs it for free -- to_dict()/save() need no new field and
        # existing cached graphs need no migration.
        self._df: Optional[Counter] = None
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def add_cue(self, cue_id: str) -> CueNode:
        return self.cues.setdefault(cue_id, CueNode(cue_id=cue_id))

    def add_content(
        self,
        content_id: str,
        text: str,
        *,
        layer: ContentLayer = "episodic",
        time: Optional[str] = None,
        topic_links: Optional[List[str]] = None,
    ) -> ContentNode:
        node = ContentNode(content_id=content_id, text=text, layer=layer, time=time,
                            topic_links=list(topic_links or []))
        self.contents[content_id] = node
        self._df = None   # corpus changed -> any cached IDF is now stale
        self._idf = {}
        return node

    def link(self, cue_id: str, tag: str, content_id: str) -> None:
        """Register a Cue-Tag-Content triple (c, g, v) in R."""
        cue = self.add_cue(cue_id)
        cue.tag_set.add(tag)
        content = self.contents[content_id]
        if tag not in content.tags:
            content.tags.append(tag)
        self.cue_tag_to_content.setdefault((cue_id, tag), set()).add(content_id)
        self.content_to_cue_tag.setdefault(content_id, set()).add((cue_id, tag))

    # ------------------------------------------------------------------
    # Traversal operators (Eq. 5, 8-9)
    # ------------------------------------------------------------------
    def cue_to_tags(self, cue_id: str) -> Set[str]:
        """phi_c->g(c): tags activated by a single cue."""
        cue = self.cues.get(cue_id)
        return set(cue.tag_set) if cue else set()

    def forward_cue_to_tag(self, active_cues: Iterable[str]) -> Set[str]:
        """Pi_c->g(C^(t)) = union_{c in C^(t)} phi_c->g(c)  (Eq. 8)."""
        tags: Set[str] = set()
        for cue_id in active_cues:
            tags |= self.cue_to_tags(cue_id)
        return tags

    def forward_tag_to_content(
        self,
        active_cues: Iterable[str],
        active_tags: Iterable[str],
        *,
        exclude_tags: Iterable[str] = (),
        exclude_content_ids: Iterable[str] = (),
    ) -> Set[str]:
        """Pi_(c,g)->v(C^(t), G^(t))  (Eq. 8).

        ``exclude_tags`` is the literal mechanism through which Planning
        experience advice like "avoid this type of tag expansion" prunes a
        branch before it is ever traversed.
        """
        exclude_tags = set(exclude_tags)
        exclude_content_ids = set(exclude_content_ids)
        contents: Set[str] = set()
        for cue_id in active_cues:
            for tag in active_tags:
                if tag in exclude_tags:
                    continue
                contents |= self.cue_tag_to_content.get((cue_id, tag), set())
        return contents - exclude_content_ids

    def reverse_content_to_cue_tag(self, active_contents: Iterable[str]) -> Set[Tuple[str, str]]:
        """Pi_v->(c,g)(V^(t))  (Eq. 9): contents activate new (cue, tag) pairs."""
        pairs: Set[Tuple[str, str]] = set()
        for content_id in active_contents:
            pairs |= self.content_to_cue_tag.get(content_id, set())
        return pairs

    def topic_to_episodes(self, topic_content_id: str) -> List[str]:
        """phi_tau->e: a topic node's constituent episodes."""
        node = self.contents.get(topic_content_id)
        return list(node.topic_links) if node else []

    # ------------------------------------------------------------------
    # Relevance ranking, for truncating an over-budget candidate set
    # (nodes.py's MAX_ACTIVE_TAGS / MAX_NEW_CONTENT_PER_ROUND) without
    # just keeping an arbitrary (e.g. alphabetical-by-id) slice of it.
    #
    # Scoring is IDF-weighted token overlap over stopword-stripped tokens,
    # not raw set-intersection size. Unweighted overlap scores every shared
    # word (an entity name or "the") as worth exactly 1 point, which lets a
    # long/generic turn out-rank the genuinely relevant one purely by
    # sharing filler words -- measured on a real LoCoMo question, this moved
    # the correct evidence turn from rank 8 to rank 4. IDF is corpus-local
    # (computed per conversation graph, not globally) and cached lazily; see
    # _ensure_idf.
    # ------------------------------------------------------------------
    def _ensure_idf(self) -> None:
        if self._df is not None:
            return
        df: Counter = Counter()
        for node in self.contents.values():
            for token in _content_tokens(node.display_text()):
                df[token] += 1
        n_docs = max(1, len(self.contents))
        self._df = df
        # BM25-style smoothed IDF, not the textbook log(N/df): plain log(N/df)
        # goes negative once a word appears in over half the corpus, which
        # would let matching a common word actively subtract from a score.
        # The +0.5 terms also avoid a divide-by-zero at df=0 or df=N.
        self._idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def _idf_of(self, token: str) -> float:
        self._ensure_idf()
        # A token absent from every content node can never appear in an
        # overlap, so its weight is unreachable in practice; 0.0 just keeps
        # the sum well-defined rather than raising KeyError.
        return self._idf.get(token, 0.0)

    def rank_tags_by_relevance(self, tags: Iterable[str], query: str) -> List[str]:
        query_tokens = _content_tokens(query)

        def _score(tag: str) -> float:
            return sum(self._idf_of(token) for token in (_content_tokens(tag) & query_tokens))

        return sorted(tags, key=lambda tag: (-_score(tag), tag))

    def rank_contents_by_relevance(self, content_ids: Iterable[str], query: str) -> List[str]:
        """Content-layer ranking. IDF-weighted, stopword-stripped token
        overlap when no embedder is attached (or when
        DISABLE_CONTENT_EMBEDDER_FUSION is set -- see below). Once
        ``attach_embedder`` has been called, this transparently fuses in
        embedding similarity via reciprocal rank fusion rather than
        replacing lexical scoring outright -- pure cosine similarity on a
        general-purpose encoder can get pulled toward topically-similar-but-
        wrong passages ("video games" broadly, instead of the one passage
        naming the actual console); fusing keeps the lexical signal's exact-
        entity precision for items it's confident about, while still letting
        semantically-related-but-token-disjoint items surface. Cue/tag layer
        ranking (``rank_tags_by_relevance``) is deliberately left without
        this fusion -- it applies only to the content layer.
        """
        content_ids = list(content_ids)
        query_tokens = _content_tokens(query)

        def _lexical_score(content_id: str) -> float:
            node = self.contents.get(content_id)
            if node is None:
                return 0.0
            # display_text(), not text: semantic_rank_contents already scores
            # the "[timestamp] ..." form via _content_embedding, and the
            # lexical half being blind to the timestamp meant a turn could be
            # discarded before the LLM ever saw it, purely because its most
            # query-relevant content was a date this scorer couldn't see.
            overlap = _content_tokens(node.display_text()) & query_tokens
            return sum(self._idf_of(token) for token in overlap)

        lexical_order = sorted(content_ids, key=lambda cid: (-_lexical_score(cid), cid))

        # Isolated-testing escape hatch: lets a validation run measure the
        # IDF change on its own, decoupled from the RRF+embedder fusion below
        # (whose interaction with a stronger lexical ranker is untested).
        # Unset (the default) leaves production behaviour identical to before
        # this flag existed.
        if os.environ.get("DISABLE_CONTENT_EMBEDDER_FUSION"):
            return lexical_order

        if not self._embedder:
            return lexical_order

        lexical_rank = {cid: i for i, cid in enumerate(lexical_order)}
        semantic_order = self.semantic_rank_contents(content_ids, query)
        semantic_rank = {cid: i for i, cid in enumerate(semantic_order)}

        rrf_k = 60  # standard reciprocal-rank-fusion damping constant
        def _fused_score(cid: str) -> float:
            return (
                1.0 / (rrf_k + lexical_rank[cid])
                + 1.0 / (rrf_k + semantic_rank.get(cid, len(content_ids)))
            )

        return sorted(content_ids, key=lambda cid: (-_fused_score(cid), cid))

    # ------------------------------------------------------------------
    # Initial cue matching (MRAgent Sec. 4.2: "extracts a set of
    # fine-grained cues and matches them against the stored cue set")
    # ------------------------------------------------------------------
    def match_query_to_cues(self, query: str, *, max_matches: int = 25) -> Set[str]:
        """A cue matches only if *every* one of its tokens appears in the
        query (a strict subset match, not "shares at least one word") --
        with hundreds of LLM-extracted multi-word cues in a real
        conversation, single-token overlap (the old `min_overlap=1`
        behaviour) matches on shared filler words alone (e.g. a cue like
        "group of bowls" matching any query containing "group"), which on
        a ~1000-cue graph fans out to hundreds of tags and contents and
        blows the LLM's context window on the very first retrieval call.
        Results are capped at ``max_matches``, preferring the most
        specific (most-token) matches when more than that many qualify.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return set()
        query_lower = query.lower()
        scored: List[Tuple[int, str]] = []
        for cue_id in self.cues:
            cue_tokens = _tokenize(cue_id)
            if not cue_tokens:
                continue
            if cue_tokens.issubset(query_tokens) or cue_id.lower() in query_lower:
                scored.append((len(cue_tokens), cue_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {cue_id for _, cue_id in scored[:max_matches]}

    # ------------------------------------------------------------------
    # Semantic matching (optional; used for document QA where the question
    # rarely contains the answer's own words, so token-overlap cue matching
    # fails). Activated only when an embedding backend is attached; otherwise
    # every method here is inert and the lexical path above is used instead.
    # ------------------------------------------------------------------
    @property
    def semantic_enabled(self) -> bool:
        return self._embedder is not None

    def attach_embedder(self, backend) -> None:
        """Give the graph an EmbeddingBackend (from iterret.experience_bank).
        Content/cue embeddings are computed lazily and cached on first use."""
        self._embedder = backend

    def _content_embedding(self, content_id: str):
        emb = self._content_emb.get(content_id)
        if emb is None:
            emb = self._embedder.encode(self.contents[content_id].display_text())
            self._content_emb[content_id] = emb
        return emb

    def _cue_embedding(self, cue_id: str):
        emb = self._cue_emb.get(cue_id)
        if emb is None:
            emb = self._embedder.encode(cue_id)
            self._cue_emb[cue_id] = emb
        return emb

    def semantic_match_cues(self, query: str, *, max_matches: int = 40,
                             min_sim: float = 0.2) -> Set[str]:
        """Cues whose embedding is most similar to the query (semantic, not
        token-overlap). Returns up to ``max_matches`` above ``min_sim``."""
        if not self._embedder or not self.cues:
            return set()
        q = self._embedder.encode(query)
        scored: List[Tuple[float, str]] = []
        for cue_id in self.cues:
            sim = self._embedder.similarity(q, self._cue_embedding(cue_id))
            if sim >= min_sim:
                scored.append((sim, cue_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {cue_id for _, cue_id in scored[:max_matches]}

    def semantic_rank_contents(self, content_ids: Iterable[str], query: str) -> List[str]:
        """Rank content ids by embedding similarity to the query (falls back to
        the lexical ranker if no embedder is attached)."""
        if not self._embedder:
            return self.rank_contents_by_relevance(content_ids, query)
        q = self._embedder.encode(query)
        return sorted(content_ids,
                      key=lambda cid: -self._embedder.similarity(q, self._content_embedding(cid)))

    def semantic_fallback_contents(self, query: str, *, top_k: int,
                                   exclude_content_ids: Iterable[str] = ()) -> List[str]:
        """Top-``top_k`` content nodes by embedding similarity to the query,
        ignoring the cue/tag graph entirely. This is the recall safety net for
        when cue-gated traversal surfaces nothing."""
        if not self._embedder:
            return []
        exclude = set(exclude_content_ids)
        q = self._embedder.encode(query)
        scored: List[Tuple[float, str]] = []
        for cid in self.contents:
            if cid in exclude:
                continue
            scored.append((self._embedder.similarity(q, self._content_embedding(cid)), cid))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cid for _, cid in scored[:top_k]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cues": {cid: {"tag_set": sorted(c.tag_set)} for cid, c in self.cues.items()},
            "contents": {
                cid: {
                    "text": c.text,
                    "layer": c.layer,
                    "time": c.time,
                    "tags": c.tags,
                    "topic_links": c.topic_links,
                }
                for cid, c in self.contents.items()
            },
            "links": sorted(
                {(cue_id, tag, content_id)
                 for (cue_id, tag), content_ids in self.cue_tag_to_content.items()
                 for content_id in content_ids}
            ),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CueTagContentGraph":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        graph = cls()
        for cid, content in data.get("contents", {}).items():
            graph.add_content(
                cid,
                content["text"],
                layer=content.get("layer", "episodic"),
                time=content.get("time"),
                topic_links=content.get("topic_links", []),
            )
        for cue_id, tag, content_id in data.get("links", []):
            graph.link(cue_id, tag, content_id)
        return graph
