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
    different section (the steps after the overview), not a copy.
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
