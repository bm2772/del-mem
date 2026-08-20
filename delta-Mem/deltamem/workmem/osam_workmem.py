"""WORKMEM: two-phase OSAM population for C-AIMMS.
Phase 1: write retrieved evidence E_T at segment (message_mean) granularity,
reusing the same ingest pattern as eval/locomo_delta.py's build_teacher_forced_snapshot.
Phase 2: generate with token granularity; S persists across both phases.
"""
import os
import re

from deltamem.core.delta_impl import (
    set_delta_mem_write_granularity,
    reset_delta_mem_states,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.eval.locomo_protocol import OFFICIAL_QA_PROMPT, OFFICIAL_MAX_NEW_TOKENS


# Detects timing questions from the QUESTION TEXT ONLY -- never from the
# dataset's ground-truth category label. That distinction matters: reading the
# question is information a real deployment also has at inference time, so
# this is not answer-key leakage (the thing fix #7 originally removed).
#
# This predicate was deleted in a refactor that argued gating was
# dataset-specific, and is restored deliberately, because the measured effects
# it gates are strongly category-dependent (see the two call sites below).
# The honest cost: this pattern is English- and wh-question-shaped, so it will
# under-fire on a differently-phrased corpus. It measures 86% recall on
# LoCoMo's own temporal-labelled subset. That is a real generalization limit,
# accepted here because the alternative -- applying both gated behaviours
# uniformly -- is measurably worse on this benchmark (again, see below).
#
# A fifth alternative, `^what\s+time\b(?!\s+management)`, was REMOVED after
# being measured across all 1540 scored questions: with its negative lookahead
# it fires on exactly 0 of them, and without the lookahead its only match is
# the single question the lookahead existed to exclude ("What time management
# techniques do Deborah and Jolene use?"). It was a dataset-specific hack
# guarding a clause that earned nothing, so deleting the pair removes a
# hardcoded LoCoMo content reference at zero measured cost -- gate coverage is
# unchanged at 309 questions and 86.0% recall on category 2.
#
# Overridable so a differently-phrased corpus can supply its own pattern
# without editing this file (the English wh-shape below is the real
# generalization limit, and an env hook is the cheapest honest escape hatch).
_TEMPORAL_QUERY_PATTERN = os.environ.get(
    "OSAM_TEMPORAL_QUERY_PATTERN",
    r"^when\b"                                             # "When did/was/will..."
    r"|\bhow\s+(?:long|old)\b"                             # duration / age
    r"|\bhow\s+many\s+(?:days?|weeks?|months?|years?)\b"   # duration counts
    r"|^(?:what|which)\s+(?:date|year|month|week|day)\b",  # "What date/year/month..."
)
_TEMPORAL_QUERY_RE = re.compile(_TEMPORAL_QUERY_PATTERN, re.IGNORECASE)


def _needs_temporal_grounding(query: str) -> bool:
    return bool(_TEMPORAL_QUERY_RE.search(query))


# Detects genuine yes/no questions from the QUESTION TEXT ONLY -- same
# no-answer-key-leakage rule as _TEMPORAL_QUERY_RE above.
#
# This exists because the previous instruction asked the MODEL to decide
# whether a question was a yes/no question ("Answer 'Yes' or 'No' only when
# the question is genuinely a yes/no question") and it demonstrably could not:
# bare "No" was emitted 113 times in run 8 on 48 "What...?" and 21 "When...?"
# questions, where it was functioning as a compressed refusal rather than an
# answer, and only 15 of those 113 had a yes/no gold. Deciding this in code
# and then giving the model only the instruction that applies removes the
# judgment call it was failing, instead of asking it more firmly.
#
# English auxiliary-verb-initial shape, which is what LoCoMo's yes/no
# questions look like ("Is Deborah married?", "Would Caroline pursue
# writing?", "Did Sam go to the gym?"). An optional short leading clause is
# allowed before the auxiliary so a preamble does not defeat the anchor
# ("Based on the conversation, did Calvin and Dave meet in Boston?").
#
# Same generalization caveat as the temporal regex: this is an English
# auxiliary-shaped heuristic and will under-fire on a differently-phrased
# corpus. Measured against all 1540 scored LoCoMo questions: fires on 40,
# 90.5% recall and 95.0% precision against questions whose gold is actually a
# yes/no judgment (including hedged golds like "Likely no", "Presumably not").
_YES_NO_AUXILIARIES = (
    r"(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would|"
    r"should|shall|may|might|must)"
)
_YES_NO_QUERY_RE = re.compile(
    r"^\s*(?:[^,?]{0,60},\s*)?" + _YES_NO_AUXILIARIES + r"\b",
    re.IGNORECASE,
)

# "Would Melanie prefer a national park or a theme park?" opens with an
# auxiliary but is an ALTERNATIVE question, not a yes/no one -- its gold is
# one of the alternatives ("National park"), so instructing the model to lead
# with Yes or No is actively wrong there. Excluding these is worth it on
# measurement, not just principle: it fixes 6 such questions at the cost of 2
# genuine yes/no questions that happen to contain "or", moving precision
# 83.3% -> 95.0% for 4.7 points of recall.
_ALTERNATIVE_QUERY_RE = re.compile(r"\bor\b", re.IGNORECASE)


def _is_yes_no_question(query: str) -> bool:
    return bool(_YES_NO_QUERY_RE.match(query)) and not _ALTERNATIVE_QUERY_RE.search(query)


# A leading "[...]" block on an evidence item, which is how a dated memory
# renders (ContentNode.display_text() prefixes "[<time>] " when the node has a
# time). Deliberately shape-based, not date-parsing: this only needs to answer
# "do these items carry a timestamp prefix at all", and any attempt to parse
# the contents would re-introduce exactly the format assumption being removed.
_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[[^\]]{3,60}\]")


def _evidence_carries_dates(session) -> bool:
    """Whether this question's evidence actually has timestamp prefixes.

    The timing instruction used to assert "Each evidence item begins with the
    date it happened" unconditionally. That is a claim about how the CALLER
    happened to build its evidence strings, and it is not always true even on
    LoCoMo: only episodic nodes carry a time, while semantic ("s*") and topic
    ("t*") nodes are created with time=None and so render with no timestamp at
    all. Asserting it regardless means that on a question retrieved entirely
    from semantic nodes -- or on any corpus that does not date its memories --
    the prompt confidently describes structure the model cannot find, which is
    the same class of error as the old "[YYYY-MM-DD | Session N]" hint that
    described a format that never existed.

    Checking the evidence instead of assuming makes the instruction
    self-verifying, and costs one regex over the messages already in hand.
    """
    messages = getattr(session, "messages", None) or []
    evidence = [
        m.get("content", "") for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if not evidence:
        return False
    dated = sum(1 for text in evidence if _TIMESTAMP_PREFIX_RE.match(text))
    # Majority rather than "any": one stray dated item among many undated ones
    # does not make "each evidence item begins with the date" a true statement.
    return dated * 2 >= len(evidence)

# Phase 1 write granularity. Maps onto the delta-mem paper's write strategies:
#   "token"        = TSW (Token-State Write)    -- what the released adapter was
#                    trained with (model card: "Variant: TSW, Write granularity:
#                    token"), so this is the ONLY setting that is train/inference
#                    consistent with models/delta-mem-adapter.
#   "message_mean" = SSW (Sequence-State Write) -- "averaging hidden states within
#                    each message, then each message updates the online state once".
#                    This is what the two-phase OSAM design calls for, but running
#                    it against a TSW checkpoint is a train/inference mismatch.
# Overridable so the two can be A/B'd on the cluster without editing this file.
PHASE1_WRITE_GRANULARITY = os.environ.get("OSAM_PHASE1_GRANULARITY", "message_mean")

# Phase 2 prompt-prefill writes. "1"/default = current behaviour (the
# instruction block and question are written into S at token granularity
# before generation starts); "0" = the prefill READS S without overwriting
# it, and only the generated tokens write.
#
# This is an UNMEASURED diagnostic arm, which is why it defaults to the
# existing behaviour rather than to the one the mechanism argues for. What it
# tests: Phase 1 writes ~12 mean-pooled evidence vectors into an 8x8 state,
# then this prefill applies ~250-330 sequential token writes on top before the
# model picks its first output token. Whether that erases Phase 1's
# contribution has never been checked. The paired granularity A/B came back
# null (message_mean vs token, diff +0.0005, CI [-0.0106, +0.0107]), and there
# are two incompatible explanations for that null -- either the prefill washes
# out Phase 1 regardless of how it was written, or OSAM's contribution is
# simply small (online_gain=0.05, delta_o_ratio still unlogged). Running with
# this set to 0 separates them: a materially different score means the prefill
# was the dominant force; an equally null result points at the second cause.
PHASE2_PROMPT_WRITE = os.environ.get("OSAM_PHASE2_PROMPT_WRITE", "1") != "0"


def populate_osam_from_evidence(session, evidence_list, *, reset=True,
                                write_granularity=None):
    """Phase 1. evidence_list: list[str], one retrieved unit per string (E_T)."""
    if reset:
        reset_delta_mem_states(session.model)

    granularity = write_granularity or PHASE1_WRITE_GRANULARITY
    set_delta_mem_write_granularity(session.model, granularity)
    # role MUST be "user", not "system". The write granularity flag is only half
    # the contract: the engine gates message_mean on a per-token write_message_ids
    # tensor, and session.py:169 stamps -1 on every SYSTEM message. With all-system
    # evidence the mask (delta_impl.py:1410, message_ids.ge(0)) is empty,
    # _message_write_inputs returns None, and delta_impl.py:2184 silently falls
    # through to the token-granularity scan -- i.e. Phase 1 degrades into Phase 2
    # while state_stats() still reports a changed S, so the degradation is
    # invisible. Upstream agrees: build_locomo_history_messages (locomo_protocol.py)
    # feeds memory content to build_teacher_forced_snapshot as "user" messages and
    # reserves "system" for the instruction only.
    session.messages = [{"role": "user", "content": unit} for unit in evidence_list]
    
    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    session._ingest_full_ids(full_ids)

    _assert_segment_write_fired(session, full_ids, len(evidence_list), granularity)
    return session.state_stats()


def _assert_segment_write_fired(session, full_ids, n_evidence: int,
                                granularity: str) -> None:
    """Fail loudly if Phase 1 did not actually write at segment granularity.

    state_stats() cannot detect this: the token-granularity fallback also
    mutates S, so a silently-degraded two-phase run looks identical to a
    correct one. These are the exact preconditions delta_impl.py checks
    before taking the message_mean branch -- assert them here, at the one
    place that can explain the failure, instead of discovering it later in
    the scores.
    """
    # "token" (TSW) writes every token unconditionally and has no segment mask to
    # verify -- there is nothing that can silently degrade, so nothing to assert.
    if n_evidence == 0 or granularity == "token":
        return

    grans = {
        getattr(m, "memory_write_granularity", None)
        for m in session.model.modules()
        if getattr(m, "memory_write_granularity", None) is not None
    }
    if grans != {granularity}:
        raise RuntimeError(
            f"OSAM Phase 1: write granularity is {grans or '{}'}, expected "
            f"{{{granularity!r}}}. set_delta_mem_write_granularity did not reach "
            "the Delta-Mem modules."
        )

    # Check the exact tensor the engine receives, not the session's own cache:
    # _runtime_write_span_ids can also fail by returning (None, None) when the
    # chat template is not prefix-stable, and delta_impl.py:1402 treats that
    # identically to an empty mask -- another silent fall-through to token writes.
    message_ids_t, _ = session._runtime_write_span_ids(full_ids)
    if message_ids_t is None:
        raise RuntimeError(
            "OSAM Phase 1: session._runtime_write_span_ids() could not recover write "
            "spans for the evidence prompt (chat template is not prefix-stable), so "
            "write_message_ids is None and delta_impl.py falls back to a "
            "token-granularity scan."
        )

    # delta_impl.py:1402/1410 -- the write mask is message_ids.ge(0), so at
    # least one evidence token must carry a non-negative message id.
    message_ids = message_ids_t.squeeze(0).tolist()
    n_written = len({mid for mid in message_ids if mid >= 0})
    if n_written == 0:
        raise RuntimeError(
            "OSAM Phase 1: every evidence token was stamped message_id=-1, so the "
            "message_mean write mask is empty and delta_impl.py falls back to a "
            "token-granularity scan. Evidence must be ingested with role='user'."
        )
    if n_written != n_evidence:
        raise RuntimeError(
            f"OSAM Phase 1: {n_evidence} evidence units produced {n_written} write "
            "segments. Each unit must be its own message so message_mean pools one "
            "write per retrieved unit."
        )

# --- Adaptive evidence narrowing (gated -- see the per-category data) ---
#
# The mechanism is dataset-agnostic: it derives a cut point from THAT
# question's own relevance-score distribution, with no category labels, no
# content patterns, and no fixed evidence-count constant. A fixed count cap
# was tried and rejected first -- it can't tell "the answer is safely in the
# top N" from "the answer is item N+1", and its value has no principled
# derivation.
#
# It is nonetheless GATED to timing questions, reversing an earlier decision
# to apply it uniformly. The uniform version was motivated by a real
# dilution signal on the temporal subset of the IDF-weighted 1540-question
# run (0-4 items -> 0.2714, 5-8 -> 0.2563, 9-12 -> 0.2393, 13-18 -> 0.2203,
# 19+ -> 0.1458; r = -0.0717, where the same measurement pre-IDF was flat at
# r = -0.0003). Those numbers are correct. But measured per category on that
# same run, the sign of the effect is NOT consistent:
#
#     MULTI_HOP    r = +0.1149   n_ev<=8: 0.1867  n_ev>8: 0.2963   (+0.110)
#     TEMPORAL     r = -0.0717   n_ev<=8: 0.2633  n_ev>8: 0.2221   (-0.041)
#     OPEN_DOMAIN  r = +0.2778   n_ev<=8: 0.1624  n_ev>8: 0.2380   (+0.076)
#     SINGLE_HOP   r = -0.0519   n_ev<=8: 0.3568  n_ev>8: 0.3410   (-0.016)
#
# Dilution holds for temporal and (weakly) single-hop, and REVERSES sharply
# for the two inferential categories, which need breadth to chain facts --
# and those two had just gained +0.017 each from the IDF retrieval fix.
# Multi-hop in particular flipped sign post-IDF (-0.0280 -> +0.1149):
# dilution is a property of BAD evidence, and better ranking partly cured it.
# Narrowing everything would risk a confirmed gain on two categories to chase
# a regression on one.
#
# Caveat kept in view: the dilution signal is correlational and this function
# has never actually been run end-to-end, so the gate also limits the blast
# radius of an unproven change.


# Adaptive temporal narrowing: OFF by default as of 2026-08-19.
#
# It was gated to timing questions, and timing is the category that has been
# losing score for four consecutive runs while its evidence was already being
# squeezed from every other direction: mean evidence per temporal question fell
# 13.70 (run 6) -> 11.06 (run 8) -> 8.89 (IDF run), and over that same period
# the 1-2-item bucket grew from n=10 to n=39 while its score fell from 0.466 to
# 0.272. Small evidence sets stopped being focus and became starvation. Adding
# a further temporal-only cut on top of that was pointing the one narrowing
# mechanism in the pipeline at the one category least able to absorb it.
#
# The justification was also weaker than the reasoning used to EXCLUDE the
# other categories from the same treatment: temporal's dilution correlation is
# r = -0.07, against r = +0.11 (multi-hop) and r = +0.28 (open-domain) used to
# argue those categories need breadth. And the function's own comment concedes
# it "has never actually been run end-to-end."
#
# Kept behind a flag rather than deleted: the dilution signal is real, and this
# is a clean single-variable experiment for a later run. Set
# WORKMEM_TEMPORAL_NARROWING=1 to restore it.
TEMPORAL_NARROWING_ENABLED = os.environ.get("WORKMEM_TEMPORAL_NARROWING", "0") == "1"


def maybe_narrow_evidence(question: str, evidence_list, encode_fn):
    """Single entry point for temporal evidence narrowing.

    Both the main eval and the A/B harness call this rather than calling
    narrow_evidence_by_score_gap directly, so the flag and the timing gate
    cannot drift apart between the two paths -- which is exactly how the
    three copies of CATEGORY_MAP ended up disagreeing.
    """
    if not TEMPORAL_NARROWING_ENABLED:
        return evidence_list
    if not _needs_temporal_grounding(question):
        return evidence_list
    return narrow_evidence_by_score_gap(question, evidence_list, encode_fn)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def narrow_evidence_by_score_gap(question: str, evidence_list, encode_fn):
    """Adaptively narrow already relevance-ranked evidence to whatever
    prefix is clearly relevant, using only that question's own relevance
    scores. Cuts at the first point where consecutive scores (recomputed
    here since filter_evidence_by_relevance discards them after sorting)
    drop by more than THIS question's own mean+stdev gap -- a boundary
    computed fresh per call, not a constant carried between questions. If
    no such gap exists, keeps everything: consistent with
    evidence_filter.py's own precedent of preferring over-inclusion to a
    forced, unjustified cut.
    """
    if not evidence_list:
        return evidence_list

    try:
        q_vec = encode_fn(question)
        scores = [_cosine(q_vec, encode_fn(item)) for item in evidence_list]
    except Exception:
        return evidence_list  # scoring failed -- don't force a cut on bad data

    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    # With exactly 2 gap values, mean+stdev is ALWAYS exactly max(gap1, gap2)
    # (provably, for any two numbers a>=b: mean=(a+b)/2, stdev=(a-b)/2, so
    # mean+stdev=a) -- meaning the strict "g > threshold" check below can
    # never fire for a 3-item list no matter how sharp the real gap is.
    # Bailing out explicitly at 3 items documents that as an intentional
    # floor, not an accidental by-product of the arithmetic.
    if len(gaps) < 3:
        return evidence_list
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std_gap = variance ** 0.5
    threshold = mean_gap + std_gap
    # The epsilon guards against floating-point subtraction noise, not
    # against real data: a perfectly uniform decline (e.g. every gap really
    # is 0.05) can still produce gaps like 0.050000000000000044 vs
    # 0.04999999999999993 purely from float representation error, and
    # without this margin that noise alone can exceed the threshold and
    # trigger a spurious cut on data with no genuine elbow at all. This is
    # a numerical-stability safeguard, not a tuned relevance parameter --
    # the same category as _cosine's "+ 1e-9" denominator guard above.
    for i, g in enumerate(gaps):
        if g > threshold + 1e-9 and g > 0:
            return evidence_list[: i + 1]

    return evidence_list  # no clear natural boundary -- keep everything


def answer_with_osam(session, query, system_instruction=None, *,
                     allow_abstention: bool = False, **gen_kwargs):
    """Generate an answer with OSAM live.

    ``allow_abstention``: when False (the default, and what the LoCoMo cats-1-4
    eval wants), the model is instructed never to decline to answer. This is a
    metric-driven choice, not a general-purpose one -- see the instruction
    comment below. Callers that DO evaluate LoCoMo category 5 (adversarial),
    where the correct answer genuinely is a refusal, must pass True;
    eval_locomo_workmem.py is the one such caller in this repo (it buckets by
    category without excluding cat 5) and would need updating before it is
    used again.
    """
    try:
        from deltamem.core.delta_impl import set_delta_mem_write_granularity
        set_delta_mem_write_granularity(session.model, "token")
    except ImportError:
        pass

    # These grounding instructions apply to every question regardless of type
    # (run 6's data: first-person suppression and the anti-refusal grounding
    # both helped broadly) -- previously they were only applied on the
    # "else" branch, so any caller-supplied system_instruction silently
    # dropped them instead of adding to them.
    #
    # The abstention instruction is deliberately a HARD prohibition rather than
    # the previous softer "don't refuse just because it isn't verbatim", and
    # that follows directly from measurement, not style:
    #   * Abstention was 28.2% of all cats-1-4 answers and 38.8% of ALL lost F1
    #     -- the single largest loss bucket in the run-8 analysis.
    #   * Abstaining is NEVER correct here: 0 of 1540 cats-1-4 gold answers is
    #     a refusal (checked directly against locomo10.json).
    #   * The payoff is therefore one-sided. Abstaining scores 0.0276 mean F1
    #     (only 6.7% of abstentions score >0 at all, via incidental token
    #     overlap); answering scores 0.3965 mean. A WRONG guess scores 0.0 --
    #     i.e. guessing risks ~0.03 and gains up to 1.0 per question.
    #   * Abstention was also measured to be uncorrelated with whether the
    #     evidence actually contains the answer (r = -0.03 vs gold-evidence
    #     presence), so this is not the model correctly detecting a gap; it
    #     has no calibration signal to preserve.
    # NOTE this is scored-benchmark-specific. On cat 5 (adversarial), or any
    # deployment where "I don't know" is a valid and desirable answer, this
    # instruction is actively wrong -- hence allow_abstention above.
    #
    # The judgment instruction now SCOPES 'Yes'/'No' rather than offering them
    # as generic worked examples. Previously they were the first two examples
    # given for any "judgment" question, and bare "No" was emitted 113 times in
    # run 8 (vs 1 in run 6) -- on 48 "What...?" and 21 "When...?" questions,
    # where it was functioning as a compressed refusal rather than an answer.
    # Only 15 of those 113 had a yes/no gold, so the examples are replaced with
    # content-shaped ones while keeping the (separately useful) "state the
    # judgment directly, don't only give reasoning" requirement.
    if allow_abstention:
        answer_policy = (
            "If the exact answer is directly stated in the evidence, use it. If it is not "
            "directly stated, use reasonable inference from the evidence and general knowledge "
            "to give your best judgment -- do not say the information is unavailable or not "
            "mentioned just because it isn't stated verbatim."
        )
    else:
        # PURELY POSITIVE PHRASING -- deliberately names none of the refusal
        # forms it is trying to suppress. The previous version enumerated them
        # ("'unknown', 'not specified', 'not mentioned', 'not available',
        # 'cannot determine', 'not applicable', 'none', 'nothing'") and the
        # model then emitted those exact strings anyway: measured on the
        # temporal subset of the IDF run, the <=2-word answers are led by
        # 'Not specified' (34), 'Cannot determine' (8) and 'Unknown' (7) --
        # every one of them a phrase the instruction had just quoted as
        # forbidden.
        #
        # That is the known failure mode of negative instructions on a small
        # backbone: quoting a phrase raises its salience rather than
        # suppressing it. The model is being shown the highest-frequency
        # refusal strings, in-context, immediately before being asked a
        # question it is uncertain about, and greedy decoding then reaches
        # for the most primed continuation. So the enumeration was not merely
        # failing to help -- it was plausibly supplying the vocabulary.
        #
        # The premise behind the old wording still holds and is what this
        # version keeps: abstaining is NEVER correct here (0 of 1540 cats-1-4
        # golds is a refusal), abstaining scores 0.0276 mean F1 against 0.3965
        # for answering, and abstention is uncorrelated with whether the
        # evidence actually contains the answer (r = -0.03), so there is no
        # calibration signal being destroyed by pushing the model to commit.
        answer_policy = (
            "Every question here has an answer. If the evidence states it, reply using those "
            "exact words. If the evidence only implies it, work out the most likely answer "
            "from the evidence and general knowledge, and give that. Commit to one specific "
            "answer even when you are not certain -- a specific answer you are unsure of is "
            "always better than a vague one."
        )

    instructions = [
        "Do not use first-person pronouns (I, me, my, we, our). Refer to people by name.",
        answer_policy,
    ]

    # EXACTLY ONE answer-shape instruction, chosen in code. Three properties
    # matter here and all three were wrong before:
    #
    #  1. Gated in code, not delegated to the model. The single previous
    #     instruction asked it to decide for itself whether a question was a
    #     yes/no question and answer accordingly; it got that classification
    #     wrong often enough that bare "No" became the most common compressed
    #     refusal (113 in run 8, on 48 "What...?" and 21 "When...?").
    #
    #  2. Mutually exclusive. Timing questions used to receive the timing
    #     instruction ON TOP of the name-a-thing one, so the prompt
    #     simultaneously demanded a calendar date and asserted "the answer
    #     must name a thing", illustrated with 'Pomodoro technique',
    #     'psychology', 'the beach' -- three examples pointing away from a
    #     date, on the category whose whole failure mode is not emitting one.
    #
    #  3. Shortest for the category that needs it most. Because the branches
    #     are exclusive, a temporal prompt now carries ~184 instruction words
    #     instead of ~226, on a 4B backbone where instructions demonstrably
    #     compete for adherence.
    #
    # Order matters: timing is checked FIRST because a timing question is
    # answered with a date whether or not it also opens with an auxiliary
    # verb ("Was it in May that Sam joined the gym?").
    #
    # The timing instruction is held back rather than appended here -- it is
    # emitted AFTER the question instead, see trailing_instruction below.
    trailing_instruction = None
    if _needs_temporal_grounding(query):
        trailing_instruction = _temporal_answer_instruction(
            evidence_carries_dates=_evidence_carries_dates(session)
        )
    elif _is_yes_no_question(query):
        instructions.append(
            "This is a yes/no question. Start with Yes or No, then add a short phrase "
            "naming the detail that settles it."
        )
    else:
        # Note this branch never writes the words yes or no, not even to
        # forbid them -- same anti-priming reasoning as answer_policy above.
        # The requirement is stated as what the answer must BE (a named
        # thing), which leaves no room for a bare judgment word without
        # having to mention one.
        instructions.append(
            "Answer by naming the specific thing being asked for (e.g. 'Pomodoro technique', "
            "'psychology', 'the beach') as a short phrase, stated directly before any "
            "elaboration. This question asks which thing, not whether something is true, so "
            "the answer must name a thing. Do not give only the supporting reasoning without "
            "ever stating the answer itself."
        )

    if system_instruction:
        instructions.append(system_instruction)

    base_prompt = OFFICIAL_QA_PROMPT.format(query)
    formatted_query = "\n".join(instructions) + "\n\n" + base_prompt

    # The timing instruction goes AFTER the question, making it the last thing
    # the model reads before it starts generating.
    #
    # This restores run 6's structure, and run 6 is the best temporal score
    # this project has recorded (0.3015, against 0.2430 for the most recent
    # full run). Run 6 built its prompt as
    #     OFFICIAL_QA_PROMPT.format(q_text + " IMPORTANT: ...timestamps...")
    # so the timing directive sat inside the question line, immediately before
    # the generation point. Every version since moved it into a leading
    # instruction block, where it now sits ~250 tokens back behind three other
    # instructions -- and absolute-date emission fell 70.4% -> 33.6% over the
    # same period.
    #
    # Two things make this worth doing on its own: instruction adherence
    # degrades with distance from the generation point on a 4B backbone, and
    # run 6 achieved its score while its format hint was FACTUALLY WRONG about
    # the timestamp format, which argues the position was carrying the result
    # rather than the wording.
    #
    # A newline is used rather than run 6's same-line concatenation: it keeps
    # the directive from reading as part of the question text while preserving
    # the property that actually matters (nothing between it and generation).
    #
    # To revert to the pre-change layout, append trailing_instruction to
    # `instructions` above instead of here.
    if trailing_instruction:
        formatted_query += "\n" + trailing_instruction

    # gen_kwargs wins if a caller passes prompt_write_enabled explicitly (the
    # A/B harness does), so the module default never silently overrides an
    # experiment arm.
    gen_kwargs.setdefault("prompt_write_enabled", PHASE2_PROMPT_WRITE)
    return session.generate_reply(formatted_query, **gen_kwargs)


def _temporal_answer_instruction(*, evidence_carries_dates: bool = True) -> str:
    """The single answer-shape instruction used for timing questions.

    ``evidence_carries_dates`` comes from inspecting the actual evidence (see
    _evidence_carries_dates). When it is False the opening sentence, which
    would otherwise assert that every evidence item begins with a date, is
    dropped -- the answer-shape half still applies, but the prompt stops
    describing structure that is not there.

    Imperative about the one operation that actually earns the score. Two
    prior versions each got half of this right. The oldest hardcoded LoCoMo's
    own "[1:56 pm on 8 May, 2023]" evidence format (which did not generalize,
    and had at one point been factually WRONG about the format). Its
    replacement genericized correctly but softened to a conditional "use them
    to give the most specific answer possible" -- dropping both the imperative
    and the NAMED operation, and applying it to all 1540 questions rather than
    the ~26% that are timing questions.

    Why naming the operation matters, measured on the IDF-weighted run: 80% of
    temporal golds (256/321) are absolute dates, and on those,
        emitted an absolute date : 39.1% of them, mean F1 0.5048
        emitted a relative expr  : 28.9% of them, mean F1 0.1344
    i.e. performing that one conversion is worth ~+0.37 F1 per question, and
    the model does it on well under half the questions that need it.

    This version also drops the enumeration, for the SAME anti-priming reason
    as answer_policy, with evidence just as direct. The previous wording spelled
    out "(yesterday, last week, last month, this year, recently)" as the
    expressions to avoid -- and the <=2-word temporal answers in that run are
    then led by 'Last week' (12), 'Next month' (9), 'Last month' (6),
    'Last year' (5), 'Last Friday' (5). The instruction was enumerating, in
    context, the exact surface forms the model went on to emit.

    The replacement states the operation positively and shows the SHAPE of a
    correct answer by example. The examples do the length-shaping a numeric
    word count cannot: on temporal questions the 3-5 word band scores 0.42-0.52
    F1 in every run while the <=2-word band scores 0.14, and the golds average
    3.35 words. Spanning the examples from '7 May 2023' down to '2018' is
    deliberate -- some golds genuinely are a bare year, so a "use N words" rule
    would push those to pad.

    Dataset caveat, stated rather than hidden: the example answers are shaped
    like LoCoMo's own temporal golds. They describe an ANSWER format, not a
    claim about how evidence is structured -- which is what the oldest version
    got factually wrong. On a corpus without dated evidence this degrades to
    inert rather than to wrong.
    """
    source = (
        "Each evidence item begins with the date it happened. Use those dates to work out "
        "the actual calendar date or period the question is asking for, and answer"
        if evidence_carries_dates else
        "Work out the actual calendar date or period the question is asking for, and answer"
    )
    return (
        f"This question asks about timing. {source} with just that date or period -- for "
        "example '7 May 2023', 'July 2023', or '2018'. Give the calendar date itself rather "
        "than describing when it happened in relation to something else."
    )
