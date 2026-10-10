"""Retrieval building blocks shared by the store, the service and the tests."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from ragkb.models import Hit


class Embedder(Protocol):
    name: str
    dims: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


def rrf(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: each list contributes 1 / (k + rank + 1) per id."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def drop_near_duplicates(hits: list[Hit]) -> list[Hit]:
    """Drop a hit whose normalized text was already kept, or whose chunk is adjacent
    (ordinal plus or minus one) to a kept chunk of the same document."""
    kept: list[Hit] = []
    seen_text: set[str] = set()
    seen_pos: set[tuple[str, int]] = set()
    for hit in hits:
        c = hit.chunk
        norm = _normalize(c.text)
        if norm in seen_text:
            continue
        if (c.doc_id, c.ordinal - 1) in seen_pos or (c.doc_id, c.ordinal + 1) in seen_pos:
            continue
        kept.append(hit)
        seen_text.add(norm)
        seen_pos.add((c.doc_id, c.ordinal))
    return kept
