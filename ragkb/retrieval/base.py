"""Retrieval building blocks shared by the store, the service, the composer and the tests.

Besides the protocols and the fusion helpers, this module holds two things the composer relies on:

- ``RankedHit``: a ``Hit`` with the retriever's notes (how relevant it looks, whether it holds an exact
  identifier from the query, which passage it was attached to as context). Where an API declares
  ``Hit`` it serializes as a plain ``Hit``, so the wire format does not change.
- The calibrated relevance bands: per model, the score under which a passage is weak (decline) and the
  score from which it is strong (answer). Between the two the composer's lexical check breaks the tie.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ragkb.models import Chunk, Hit


class Embedder(Protocol):
    """Vectors for passages and queries, plus the model's own token count and passage window, so the
    chunker keeps every passage inside what the model reads (a longer one is embedded truncated)."""

    name: str
    dims: int
    max_tokens: int  # the most tokens one passage can carry without truncation

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


Relevance = Literal["strong", "borderline", "weak"]
_LEVEL_ORDER: dict[str, int] = {"weak": 0, "borderline": 1, "strong": 2}


class RankedHit(Hit):
    """A Hit with the retriever's notes for the composer.

    - ``relevance``: the verdict of the calibrated scores (rerank score, cosine), None when no score
      of this hit has a calibrated band (an unknown model, or a passage attached as context).
    - ``exact_match``: the passage holds, verbatim, an identifier or quoted phrase from the query that
      few passages contain (``SKU-04417``, ``FH-MNT-01``, ``"core hours"``).
    - ``neighbour_of``: set on a passage attached for context; the chunk id of the passage it continues.
      It was not ranked, so its ``score`` is 0 and it never decides whether to answer.
    """

    relevance: Relevance | None = None
    exact_match: bool = False
    neighbour_of: str | None = None


def is_neighbour(hit: Hit) -> bool:
    return getattr(hit, "neighbour_of", None) is not None


def ranked_only(hits: Iterable[Hit]) -> list[Hit]:
    """The hits that were ranked, without the passages attached as context."""
    return [hit for hit in hits if not is_neighbour(hit)]


# ---------------------------------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------------------------------


def rrf(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: each list contributes 1 / (k + rank + 1) per id."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# A full-text hit with at least this share of the best BM25 score keeps its full weight in fusion.
LEXICAL_FULL_WEIGHT_SHARE = 0.5


def lexical_weights(scores: Sequence[float | None]) -> list[float]:
    """Weights for the full-text list in rank fusion, from each hit's BM25 score over the best one.

    BM25 already weighs every query term by its rarity in the collection. A hit with at least
    LEXICAL_FULL_WEIGHT_SHARE of the best score matched the question's distinctive words and keeps the
    full weight, so ordinary questions fuse as plain reciprocal rank fusion. A hit far below that matched
    only common words (the "SKU" header repeated in every chunk of a price list, against the one chunk
    that also holds "04417") and its weight falls linearly toward zero. Without usable scores every
    weight is 1.
    """
    values = [float(s) if s is not None and s > 0 else 0.0 for s in scores]
    top = max(values, default=0.0)
    if top <= 0:
        return [1.0] * len(values)
    return [min(1.0, value / (top * LEXICAL_FULL_WEIGHT_SHARE)) for value in values]


# ---------------------------------------------------------------------------------------------------
# Exact terms: identifiers and quoted phrases from the query
# ---------------------------------------------------------------------------------------------------

_QUOTED = re.compile(r"\"([^\"\n]{3,80})\"|“([^”\n]{3,80})”")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-/._][A-Za-z0-9]+)*")
_DIGIT_GROUPS = re.compile(r"\d[-/]\d")
_PARTS = re.compile(r"[A-Za-z]+|\d+")


def _is_identifier(token: str) -> bool:
    """Mixed letters and digits (SKU-04417, FH-MNT-01, x86), or digit groups joined by a dash or a
    slash (2026-12-28, 4471/2026). Plain words and plain numbers are not identifiers."""
    if len(token) < 3:
        return False
    letters = any(ch.isalpha() for ch in token)
    digits = any(ch.isdigit() for ch in token)
    if letters and digits:
        return True
    return digits and not letters and len(token) >= 5 and bool(_DIGIT_GROUPS.search(token))


def exact_terms(query: str) -> list[str]:
    """The identifier-shaped tokens and the quoted phrases of *query*, in order, without repeats."""
    terms: list[str] = []
    for match in _QUOTED.finditer(query or ""):
        phrase = " ".join((match.group(1) or match.group(2) or "").split())
        if len(phrase) >= 3 and phrase.casefold() not in (t.casefold() for t in terms):
            terms.append(phrase)
    for match in _TOKEN.finditer(query or ""):
        token = match.group(0)
        if _is_identifier(token) and token.casefold() not in (t.casefold() for t in terms):
            terms.append(token)
    return terms


def exact_pattern(term: str) -> re.Pattern[str]:
    """A case-insensitive pattern for *term* as a whole token. An identifier also matches with another
    separator or none between its parts (SKU-04417, SKU 04417, sku04417); a phrase matches across any
    whitespace."""
    if " " in term:
        body = r"\s+".join(re.escape(word) for word in term.split())
    else:
        body = r"[-/._\s]?".join(re.escape(part) for part in _PARTS.findall(term))
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------
# Near-duplicates
# ---------------------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")

# How much of a chunk's end is compared with its neighbour to detect the chunker's overlap.
OVERLAP_PROBE_CHARS = 120


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _overlaps(a: str, b: str) -> bool:
    """True when the last OVERLAP_PROBE_CHARS normalized characters of one text appear in the other."""
    if not a or not b:
        return False
    return a[-OVERLAP_PROBE_CHARS:] in b or b[-OVERLAP_PROBE_CHARS:] in a


def drop_near_duplicates(hits: list[Hit]) -> list[Hit]:
    """Drop a hit whose normalized text was already kept, or whose chunk is adjacent (ordinal plus or
    minus one) to a kept chunk of the same document AND repeats its text (the chunker's overlap).

    An adjacent chunk with its own text is kept: with structure-aware chunks a neighbour is usually a
    different section (the steps after the overview), not a copy. Passages attached as context
    (neighbour expansion) are added after this step and never pass through it.
    """
    kept: list[Hit] = []
    seen_text: set[str] = set()
    kept_at: dict[tuple[str, int], str] = {}
    for hit in hits:
        c = hit.chunk
        norm = _normalize(c.text)
        if norm in seen_text:
            continue
        neighbours = (kept_at.get((c.doc_id, c.ordinal - 1)), kept_at.get((c.doc_id, c.ordinal + 1)))
        if any(other is not None and _overlaps(norm, other) for other in neighbours):
            continue
        kept.append(hit)
        seen_text.add(norm)
        kept_at[(c.doc_id, c.ordinal)] = norm
    return kept


# ---------------------------------------------------------------------------------------------------
# Relevance bands
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """A score below ``weak_below`` is weak, a score at or above ``strong_from`` is strong, and the
    scores in between are borderline: there the composer's lexical check decides."""

    weak_below: float
    strong_from: float

    def level(self, score: float) -> Relevance:
        if score < self.weak_below:
            return "weak"
        if score >= self.strong_from:
            return "strong"
        return "borderline"


DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Calibrated on the sample golden set (samples/golden.yaml) plus 24 extra paraphrased and out-of-scope
# questions over the same corpus. Best cross-encoder score (a logit): the in-scope questions scored
# -5.8 to 9.8 and the out-of-scope ones -11.0 to -1.1, with the near miss "a dress code at the Port
# Sallow office" at -1.1. Best cosine of BAAI/bge-small-en-v1.5: in scope 0.49 to 0.90, out of scope
# 0.47 to 0.65 (the same near miss at 0.652, "money for working from home" at 0.664). Neither score
# separates every case alone, so a hit is strong when either score is strong, weak only when every
# calibrated score is weak, and borderline otherwise. Keys are model names, compared without case.
RERANK_BANDS: dict[str, Band] = {DEFAULT_RERANK_MODEL.casefold(): Band(weak_below=-7.0, strong_from=0.0)}
COSINE_BANDS: dict[str, Band] = {"baai/bge-small-en-v1.5": Band(weak_below=0.60, strong_from=0.66)}


def rerank_band(model_name: str | None) -> Band | None:
    return RERANK_BANDS.get((model_name or "").strip().casefold())


def cosine_band(model_name: str | None) -> Band | None:
    return COSINE_BANDS.get((model_name or "").strip().casefold())


def classify(
    hit: Hit, *, rerank: Band | None, cosine: Band | None, exact_match: bool = False
) -> Relevance | None:
    """The hit's level from its calibrated scores: strong when any score is strong (or the hit holds a
    rare exact term from the query), weak when every calibrated score is weak, borderline otherwise;
    None when none of its scores has a band."""
    levels: list[Relevance] = []
    if rerank is not None and hit.rerank_score is not None:
        levels.append(rerank.level(hit.rerank_score))
    if cosine is not None and hit.vector_score is not None:
        levels.append(cosine.level(hit.vector_score))
    if exact_match:
        levels.append("strong")
    if not levels:
        return None
    return max(levels, key=lambda level: _LEVEL_ORDER[level])


def relevance_of(hit: Hit) -> Relevance | None:
    """The retriever's verdict on *hit*. A plain Hit (built by a caller, not the retriever) with a
    rerank score is judged by the default reranker's band. None for a passage attached as context and
    for a hit with no calibrated score."""
    if is_neighbour(hit):
        return None
    if isinstance(hit, RankedHit):
        return hit.relevance
    if hit.rerank_score is not None:
        return RERANK_BANDS[DEFAULT_RERANK_MODEL.casefold()].level(hit.rerank_score)
    return None


def best_relevance(hits: Iterable[Hit]) -> Relevance | None:
    """The best level among the ranked hits, or None when no hit has one."""
    levels = [level for level in (relevance_of(hit) for hit in hits) if level is not None]
    if not levels:
        return None
    return max(levels, key=lambda level: _LEVEL_ORDER[level])


def passage_chars(chunk: Chunk) -> int:
    """An estimate of the characters a passage takes in the composition prompt: its text plus a header
    with the title and section."""
    return len((chunk.text or "").strip()) + len(chunk.title or "") + len(chunk.section or "") + 32
