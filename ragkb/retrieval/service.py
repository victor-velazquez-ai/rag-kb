"""Retriever: one query in, ranked passages out.

1. Candidates: the CANDIDATES best passages by vector (the minimum score applies here only) and the
   CANDIDATES best by full text (BM25).
2. Fusion by reciprocal rank (k=60). The full-text side is weighted by term rarity (each hit's BM25 over
   the best BM25), so near-identical chunks that only share a common word ("SKU" in every row of a
   price list) no longer outvote the one chunk that matches the rare word. A passage that holds,
   verbatim, an identifier from the query (SKU-04417, FH-MNT-01) or a quoted phrase, when at most
   EXACT_MAX_CHUNKS candidates hold it, is boosted above every other passage.
3. Near-duplicates dropped, then the optional cross-encoder reranks the whole pool; exact-term passages
   stay first.
4. Every returned passage carries its cosine (computed for passages found by full text only) and its
   relevance verdict from the calibrated bands in `ragkb.retrieval.base`.
5. Selection. Without a budget: the first top_k (the search API). With a budget (the composer's
   MAX_CONTEXT_CHARS): at least top_k, then more passages in rank order while they fit and are not weak,
   then, with neighbour expansion, the previous and next chunk of the top one or two passages when they
   fit, placed right after their anchor and marked with ``neighbour_of``.

Degradation: when full-text search fails (for example the index is missing), the search continues on
vectors alone and logs why. When the embedder or the vector search fails, RetrievalUnavailable is
raised, which callers turn into the "search unavailable" state. A failing reranker or neighbour lookup
never fails the search. Logs carry counts and error types, never passage text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ragkb.config import Settings
from ragkb.models import Chunk, Hit
from ragkb.retrieval.base import (
    Band,
    Embedder,
    RankedHit,
    Relevance,
    Reranker,
    classify,
    cosine_band,
    drop_near_duplicates,
    exact_pattern,
    exact_terms,
    lexical_weights,
    passage_chars,
    rerank_band,
)
from ragkb.store.base import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60
# An exact identifier or quoted phrase boosts a passage only when this few candidates contain it.
EXACT_MAX_CHUNKS = 3
# Larger than anything fusion can give (two lists at rank one), so an exact match always comes first.
EXACT_BOOST = 2.0 / (RRF_K + 1)
# Neighbour expansion: the passages whose previous and next chunks are attached, and how far to reach.
NEIGHBOUR_ANCHORS = 2
NEIGHBOUR_SPAN = 1
# The blank line between two passages in the prompt.
SEPARATOR_CHARS = 2


class RetrievalUnavailable(RuntimeError):
    """Search cannot run right now (store or embedder failure). The message is safe to show."""


def _as_ranked(hit: Hit) -> RankedHit:
    if isinstance(hit, RankedHit):
        return hit
    return RankedHit(**{name: getattr(hit, name) for name in Hit.model_fields})


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


class Retriever:
    """Build once and share: it holds no per-request state."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        min_score: float = 0.0,
        *,
        candidates: int = 30,
        neighbour_expansion: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.candidates = max(1, int(candidates))
        self.neighbour_expansion = bool(neighbour_expansion)
        self._rerank_band: Band | None = rerank_band(getattr(reranker, "name", None)) if reranker else None
        self._cosine_band: Band | None = cosine_band(getattr(embedder, "name", None))

    @classmethod
    def from_settings(
        cls, store: VectorStore, embedder: Embedder, settings: Settings, reranker: Reranker | None = None
    ) -> Retriever:
        """The retriever the app uses: MIN_SCORE, CANDIDATES and NEIGHBOUR_EXPANSION from *settings*."""
        return cls(
            store,
            embedder,
            reranker,
            settings.MIN_SCORE,
            candidates=settings.CANDIDATES,
            neighbour_expansion=settings.NEIGHBOUR_EXPANSION,
        )

    # ------------------------------------------------------------------------------------- search

    def search(
        self, collection_id: str, query: str, top_k: int, *, budget_chars: int | None = None
    ) -> list[Hit]:
        """Ranked passages for *query*. Without *budget_chars*: the top_k best. With it (pass the
        composer's MAX_CONTEXT_CHARS when the passages go to compose): at least top_k, more while they
        fit, plus neighbours of the top passages. Every hit is a ``RankedHit``."""
        query = query.strip()
        if not query or top_k <= 0:
            return []
        pool = max(self.candidates, top_k)

        try:
            query_vector = self.embedder.embed_query(query)
            vector_hits = self.store.vector_search(collection_id, query_vector, pool, min_score=self.min_score)
        except Exception as exc:  # every failure here means search is down
            log.error(
                "vector search failed for collection %s: %s: %s", collection_id, type(exc).__name__, exc
            )
            raise RetrievalUnavailable("Search is unavailable right now.") from exc

        try:
            text_hits = self.store.text_search(collection_id, query, pool)
        except Exception as exc:  # noqa: BLE001 - degrade to vector-only search
            log.warning("full-text search failed, using vector search only: %s: %s", type(exc).__name__, exc)
            text_hits = []

        ranked = self._rank(query, vector_hits, text_hits, pool)
        if budget_chars is None:
            selected: list[RankedHit] = ranked[:top_k]
        else:
            selected = self._fit(ranked, top_k, budget_chars)
            if self.neighbour_expansion:
                selected = self._expand(selected, budget_chars)
        selected = self._fill_cosines(query_vector, selected)
        selected = [self._annotate(hit) for hit in selected]
        log.debug(
            "search: vector=%d text=%d ranked=%d returned=%d exact=%d neighbours=%d",
            len(vector_hits), len(text_hits), len(ranked), len(selected),
            sum(1 for h in selected if h.exact_match), sum(1 for h in selected if h.neighbour_of),
        )
        return list(selected)

    # ------------------------------------------------------------------------------------ ranking

    def _rank(self, query: str, vector_hits: list[Hit], text_hits: list[Hit], pool: int) -> list[RankedHit]:
        fused = self._fuse(vector_hits, text_hits, exact_terms(query))
        candidates = [_as_ranked(h) for h in drop_near_duplicates(list(fused))[:pool]]
        if self.reranker is None or not candidates:
            return candidates
        reranked = [_as_ranked(h) for h in self.reranker.rerank(query, list(candidates), len(candidates))]
        # The cross-encoder orders the pool; a rare exact identifier the person typed still comes first.
        return [h for h in reranked if h.exact_match] + [h for h in reranked if not h.exact_match]

    @staticmethod
    def _fuse(vector_hits: list[Hit], text_hits: list[Hit], terms: list[str]) -> list[RankedHit]:
        """Weighted reciprocal rank fusion plus the exact-term boost; ties go to the better vector rank,
        then the chunk id."""
        by_id: dict[str, Hit] = {}
        vector_rank: dict[str, int] = {}
        vector_score: dict[str, float | None] = {}
        for rank, hit in enumerate(vector_hits):
            cid = hit.chunk.id
            if cid in vector_rank:
                continue
            by_id.setdefault(cid, hit)
            vector_rank[cid] = rank
            vector_score[cid] = hit.vector_score if hit.vector_score is not None else hit.score

        raw = [h.lexical_score if h.lexical_score is not None else h.score for h in text_hits]
        weights = lexical_weights(raw)
        lexical_rank: dict[str, int] = {}
        lexical_score: dict[str, float | None] = {}
        lexical_weight: dict[str, float] = {}
        for index, hit in enumerate(text_hits):
            cid = hit.chunk.id
            if cid in lexical_rank:
                continue
            by_id.setdefault(cid, hit)
            lexical_rank[cid] = index + 1
            lexical_score[cid] = raw[index]
            lexical_weight[cid] = weights[index]

        scores: dict[str, float] = dict.fromkeys(by_id, 0.0)
        for cid, rank in vector_rank.items():
            scores[cid] += 1.0 / (RRF_K + rank + 1)
        for cid, rank in lexical_rank.items():
            scores[cid] += lexical_weight[cid] / (RRF_K + rank)
        exact = Retriever._exact_matches(by_id, terms)
        for cid, matched in exact.items():
            scores[cid] += matched * EXACT_BOOST

        ordered = sorted(scores, key=lambda cid: (-scores[cid], vector_rank.get(cid, len(vector_hits)), cid))
        return [
            RankedHit(
                chunk=by_id[cid].chunk,
                score=scores[cid],
                vector_score=vector_score.get(cid),
                lexical_rank=lexical_rank.get(cid),
                lexical_score=lexical_score.get(cid),
                rerank_score=None,
                exact_match=cid in exact,
            )
            for cid in ordered
        ]

    @staticmethod
    def _exact_matches(by_id: dict[str, Hit], terms: list[str]) -> dict[str, int]:
        """Chunk id -> how many of the query's exact terms it holds, counting only terms that at most
        EXACT_MAX_CHUNKS candidates hold (a term found everywhere tells nothing)."""
        matched: dict[str, int] = {}
        for term in terms:
            pattern = exact_pattern(term)
            holders = [cid for cid, hit in by_id.items() if pattern.search(hit.chunk.text or "")]
            if 1 <= len(holders) <= EXACT_MAX_CHUNKS:
                for cid in holders:
                    matched[cid] = matched.get(cid, 0) + 1
        return matched

    # ---------------------------------------------------------------------------------- selection

    def _level(self, hit: RankedHit) -> Relevance | None:
        return classify(hit, rerank=self._rerank_band, cosine=self._cosine_band, exact_match=hit.exact_match)

    def _fit(self, ranked: list[RankedHit], top_k: int, budget_chars: int) -> list[RankedHit]:
        """At least top_k passages, then more in rank order while they fit the budget; past top_k a
        passage the calibrated scores call weak is left out."""
        out: list[RankedHit] = []
        used = 0
        for hit in ranked:
            cost = passage_chars(hit.chunk) + (SEPARATOR_CHARS if out else 0)
            if len(out) >= top_k:
                if used + cost > budget_chars:
                    break
                if self._level(hit) == "weak":
                    continue
            out.append(hit)
            used += cost
        return out

    def _expand(self, selected: list[RankedHit], budget_chars: int) -> list[RankedHit]:
        """Attach the previous and next chunk of the top NEIGHBOUR_ANCHORS passages when they fit what is
        left of the budget, so a table or a numbered procedure split across chunks comes through whole.
        Each is placed right after its anchor, with score 0 and ``neighbour_of`` set."""
        used = sum(passage_chars(h.chunk) for h in selected) + SEPARATOR_CHARS * max(0, len(selected) - 1)
        have = {h.chunk.id for h in selected}
        anchors = [h for h in selected if self._level(h) != "weak"][:NEIGHBOUR_ANCHORS]
        attached: dict[str, list[RankedHit]] = {}
        for anchor in anchors:
            for chunk in self._neighbour_chunks(anchor.chunk):
                if chunk.id in have:
                    continue
                cost = passage_chars(chunk) + SEPARATOR_CHARS
                if used + cost > budget_chars:
                    continue
                attached.setdefault(anchor.chunk.id, []).append(
                    RankedHit(chunk=chunk, score=0.0, neighbour_of=anchor.chunk.id)
                )
                have.add(chunk.id)
                used += cost
        if not attached:
            return selected
        out: list[RankedHit] = []
        for hit in selected:
            out.append(hit)
            out.extend(sorted(attached.get(hit.chunk.id, []), key=lambda h: h.chunk.ordinal))
        return out

    def _neighbour_chunks(self, chunk: Chunk) -> list[Chunk]:
        """The chunks of the same document within NEIGHBOUR_SPAN ordinals of *chunk*, without it."""
        found: list[Chunk] | None = None
        finder: Any = getattr(self.store, "neighbours", None)
        try:
            if callable(finder):
                try:
                    found = list(finder(chunk.doc_id, chunk.ordinal, NEIGHBOUR_SPAN) or [])
                except (AttributeError, NotImplementedError):
                    found = None
            if found is None:
                found = list(self.store.chunks_for(chunk.doc_id))
        except Exception as exc:  # noqa: BLE001 - context passages are optional
            log.warning("neighbour lookup failed, continuing without context passages: %s: %s",
                        type(exc).__name__, exc)
            return []
        return sorted(
            (
                c for c in found
                if c.doc_id == chunk.doc_id
                and c.collection_id == chunk.collection_id
                and c.ordinal != chunk.ordinal
                and abs(c.ordinal - chunk.ordinal) <= NEIGHBOUR_SPAN
            ),
            key=lambda c: c.ordinal,
        )

    # --------------------------------------------------------------------------------- annotation

    def _fill_cosines(self, query_vector: list[float], hits: list[RankedHit]) -> list[RankedHit]:
        """Give every passage its cosine: those found by full text only (and context passages) are
        embedded again here, which yields exactly the stored vector."""
        missing = [i for i, hit in enumerate(hits) if hit.vector_score is None]
        if not missing:
            return hits
        try:
            vectors = self.embedder.embed([hits[i].chunk.embed_text or hits[i].chunk.text for i in missing])
        except Exception as exc:  # noqa: BLE001 - the score is informative only
            log.warning("could not compute cosines for %d passages: %s: %s", len(missing), type(exc).__name__, exc)
            return hits
        out = list(hits)
        for i, vector in zip(missing, vectors, strict=False):
            out[i] = out[i].model_copy(update={"vector_score": _dot(query_vector, vector)})
        return out

    def _annotate(self, hit: RankedHit) -> RankedHit:
        if hit.neighbour_of is not None:
            return hit
        return hit.model_copy(update={"relevance": self._level(hit)})
