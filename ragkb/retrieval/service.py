"""Retriever: one query in, ranked passages out.

Hybrid search: vector top_k*4 (the minimum score applies here only) and full-text top_k*4, fused by
reciprocal rank (k=60) to top_k*3, near-duplicates dropped, optionally reranked, sliced to top_k.

Degradation: when full-text search fails (for example the index is missing), the search continues on
vectors alone and logs why. When the embedder or the vector search fails, RetrievalUnavailable is
raised, which callers turn into the "search unavailable" state.
"""

from __future__ import annotations

import logging

from ragkb.models import Hit
from ragkb.retrieval.base import Embedder, Reranker, drop_near_duplicates, rrf
from ragkb.store.base import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60


class RetrievalUnavailable(RuntimeError):
    """Search cannot run right now (store or embedder failure). The message is safe to show."""


class Retriever:
    """Build once and share: it holds no per-request state."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        min_score: float = 0.0,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.min_score = max(0.0, min(1.0, float(min_score)))

    def search(self, collection_id: str, query: str, top_k: int) -> list[Hit]:
        query = query.strip()
        if not query or top_k <= 0:
            return []
        pool = top_k * 4

        try:
            query_vector = self.embedder.embed_query(query)
            vector_hits = self.store.vector_search(
                collection_id, query_vector, pool, min_score=self.min_score
            )
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

        fused = self._fuse(vector_hits, text_hits)[: top_k * 3]
        candidates = drop_near_duplicates(fused)
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    @staticmethod
    def _fuse(vector_hits: list[Hit], text_hits: list[Hit]) -> list[Hit]:
        """Merge both lists by reciprocal rank; ties go to the better vector rank, then the chunk id."""
        scores = rrf([[h.chunk.id for h in vector_hits], [h.chunk.id for h in text_hits]], k=RRF_K)
        by_id: dict[str, Hit] = {}
        vector_rank: dict[str, int] = {}
        for rank, hit in enumerate(vector_hits):
            by_id.setdefault(hit.chunk.id, hit)
            vector_rank.setdefault(hit.chunk.id, rank)
        lexical_rank: dict[str, int] = {}
        for rank, hit in enumerate(text_hits, start=1):
            by_id.setdefault(hit.chunk.id, hit)
            lexical_rank.setdefault(hit.chunk.id, rank)
        vector_score = {h.chunk.id: h.vector_score for h in vector_hits}
        ordered = sorted(scores, key=lambda cid: (-scores[cid], vector_rank.get(cid, len(vector_hits)), cid))
        return [
            Hit(
                chunk=by_id[cid].chunk,
                score=scores[cid],
                vector_score=vector_score.get(cid),
                lexical_rank=lexical_rank.get(cid),
                rerank_score=None,
            )
            for cid in ordered
        ]
