"""Scorers for a golden-set run.

Positives and negatives are scored separately and never mixed:

- recall@k and MRR over positives: a positive is a hit at rank r when one of its expected sources is
  the document of the r-th retrieved passage or of the r-th citation (the better of the two ranks).
- answer rate over positives: answered instead of declined.
- citation rate over answered positives: at least one citation.
- keyword rate over positives that list `expect_keywords`: every keyword appears in the answer text.
- false-answer rate over negatives: came back `answered`. Decline rate over negatives: came back
  `no_answer` or `no_documents`.

A case that could not run (search or provider unavailable, rate limited, an error) is "unmeasured":
it is left out of every rate and counted on its own. A metric with nothing to measure is `None`,
and a gate that measured too little reports SKIPPED, never PASS.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

ANSWERED = "answered"
DECLINED_STATUSES = frozenset({"no_answer", "no_documents"})
UNMEASURED_STATUSES = frozenset({"provider_unavailable", "retrieval_unavailable", "rate_limited"})

RECALL_GATE_K = 5
RECALL_GATE = 0.85
FALSE_ANSWER_GATE = 0.05
MIN_RAN_FRACTION = 0.5

EXIT_CODES = {"PASS": 0, "FAIL": 1, "SKIPPED": 2}


@dataclass
class CaseResult:
    """What one golden case produced. `retrieved` and `cited` hold, in rank order, the names that
    identify each passage's document (its title and file name)."""

    case_id: str
    kind: Literal["positive", "negative"]
    question: str = ""
    expect_sources: list[str] = field(default_factory=list)
    expect_keywords: list[str] = field(default_factory=list)
    status: str | None = None
    unmeasured_reason: str | None = None
    retrieved: list[list[str]] = field(default_factory=list)
    cited: list[list[str]] = field(default_factory=list)
    answer_text: str = ""
    elapsed_ms: int = 0
    model: str | None = None

    @property
    def measured(self) -> bool:
        return self.unmeasured_reason is None and self.status is not None and self.status not in UNMEASURED_STATUSES

    @property
    def answered(self) -> bool:
        return self.status == ANSWERED

    @property
    def declined(self) -> bool:
        return self.status in DECLINED_STATUSES

    @property
    def citation_count(self) -> int:
        return len(self.cited)


# ---------------------------------------------------------------- matching


def _variants(name: str) -> set[str]:
    value = name.strip().replace("\\", "/").casefold()
    if not value:
        return set()
    out = {value}
    base = PurePosixPath(value).name
    out.add(base)
    if "." in base:
        out.add(PurePosixPath(base).stem)
    return out


def source_matches(expected: Iterable[str], names: Iterable[str]) -> bool:
    """True when an expected source names the document: same title or file name, ignoring case,
    folders and the file extension."""
    wanted: set[str] = set()
    for item in expected:
        wanted |= _variants(item)
    have: set[str] = set()
    for item in names:
        have |= _variants(item)
    return bool(wanted & have)


def _first_rank(expected: Sequence[str], ranked: Sequence[Sequence[str]]) -> int | None:
    for rank, names in enumerate(ranked, start=1):
        if source_matches(expected, names):
            return rank
    return None


def hit_rank(result: CaseResult) -> int | None:
    """1-based rank of the first expected source among the retrieved passages or the citations."""
    ranks = [r for r in (_first_rank(result.expect_sources, result.retrieved),
                         _first_rank(result.expect_sources, result.cited)) if r is not None]
    return min(ranks) if ranks else None


def missing_keywords(result: CaseResult) -> list[str]:
    text = result.answer_text.casefold()
    return [k for k in result.expect_keywords if k.casefold() not in text]


def case_outcome(result: CaseResult) -> str:
    """One word for the report row: ok, miss, declined, false_answer or unmeasured."""
    if not result.measured:
        return "unmeasured"
    if result.kind == "negative":
        return "false_answer" if result.answered else "ok"
    rank = hit_rank(result)
    if rank is None or rank > RECALL_GATE_K:
        return "miss"
    return "ok" if result.answered else "declined"


# ---------------------------------------------------------------- totals and gate


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def recall_at(results: Sequence[CaseResult], k: int) -> float | None:
    positives = [r for r in results if r.kind == "positive" and r.measured]
    hits = sum(1 for r in positives if (rank := hit_rank(r)) is not None and rank <= k)
    return _rate(hits, len(positives))


def mean_reciprocal_rank(results: Sequence[CaseResult]) -> float | None:
    positives = [r for r in results if r.kind == "positive" and r.measured]
    if not positives:
        return None
    total = sum(1.0 / rank for r in positives if (rank := hit_rank(r)) is not None)
    return round(total / len(positives), 4)


def score(results: Sequence[CaseResult], top_k: int | None = None) -> dict[str, Any]:
    """Totals over a run: counts, recall@1/3/5 (and @top_k), MRR and the answer, citation, keyword,
    false-answer and decline rates. Rates are `None` when there was nothing to measure."""
    positives = [r for r in results if r.kind == "positive"]
    negatives = [r for r in results if r.kind == "negative"]
    pos_ran = [r for r in positives if r.measured]
    neg_ran = [r for r in negatives if r.measured]
    answered_pos = [r for r in pos_ran if r.answered]
    keyword_cases = [r for r in pos_ran if r.expect_keywords]

    ks = sorted({1, 3, RECALL_GATE_K, *([top_k] if top_k else [])})
    totals: dict[str, Any] = {
        "cases": len(results),
        "positives": len(positives),
        "negatives": len(negatives),
        "ran": len(pos_ran) + len(neg_ran),
        "unmeasured": len(results) - len(pos_ran) - len(neg_ran),
        "positives_ran": len(pos_ran),
        "negatives_ran": len(neg_ran),
    }
    for k in ks:
        totals[f"recall_at_{k}"] = recall_at(results, k)
    totals["mrr"] = mean_reciprocal_rank(results)
    totals["answer_rate"] = _rate(len(answered_pos), len(pos_ran))
    totals["citation_rate"] = _rate(sum(1 for r in answered_pos if r.citation_count > 0), len(answered_pos))
    totals["keyword_rate"] = _rate(sum(1 for r in keyword_cases if r.answered and not missing_keywords(r)),
                                   len(keyword_cases))
    totals["false_answer_rate"] = _rate(sum(1 for r in neg_ran if r.answered), len(neg_ran))
    totals["decline_rate"] = _rate(sum(1 for r in neg_ran if r.declined), len(neg_ran))
    return totals


def gate(totals: dict[str, Any]) -> dict[str, Any]:
    """PASS, FAIL or SKIPPED, with the reasons and the exit code for the CLI (0, 1, 2).

    SKIPPED when fewer than half the cases ran, or when recall or the false-answer rate could not be
    measured at all. Otherwise FAIL when recall@5 is under 0.85 or the false-answer rate is over 0.05.
    """
    thresholds = {
        f"recall_at_{RECALL_GATE_K}_min": RECALL_GATE,
        "false_answer_rate_max": FALSE_ANSWER_GATE,
        "min_ran_fraction": MIN_RAN_FRACTION,
    }
    reasons: list[str] = []
    cases = totals.get("cases", 0)
    ran = totals.get("ran", 0)
    recall = totals.get(f"recall_at_{RECALL_GATE_K}")
    false_answers = totals.get("false_answer_rate")

    if cases == 0 or ran < cases * MIN_RAN_FRACTION:
        reasons.append(f"only {ran} of {cases} cases ran; at least half must run to judge")
    if recall is None:
        reasons.append("no positive case ran, so recall was not measured")
    if false_answers is None:
        reasons.append("no negative case ran, so the false-answer rate was not measured")
    if reasons:
        outcome = "SKIPPED"
    else:
        if recall < RECALL_GATE:
            reasons.append(f"recall@{RECALL_GATE_K} {recall:.2f} is under {RECALL_GATE:.2f}")
        if false_answers > FALSE_ANSWER_GATE:
            reasons.append(f"false-answer rate {false_answers:.2f} is over {FALSE_ANSWER_GATE:.2f}")
        outcome = "FAIL" if reasons else "PASS"
        if outcome == "PASS":
            reasons.append(
                f"recall@{RECALL_GATE_K} {recall:.2f} >= {RECALL_GATE:.2f} and "
                f"false-answer rate {false_answers:.2f} <= {FALSE_ANSWER_GATE:.2f}"
            )
    return {"outcome": outcome, "exit_code": EXIT_CODES[outcome], "reasons": reasons, "thresholds": thresholds}
