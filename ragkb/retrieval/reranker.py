"""Rerankers: a local cross-encoder through fastembed, and a no-op.

A reranker never breaks search: on any error it returns the hits in the order it received them.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragkb.models import Hit

log = logging.getLogger(__name__)

_RETRY_LOAD_AFTER_S = 600.0


class NoReranker:
    """Keeps the fused order."""

    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return list(hits[:top_n])


class LocalReranker:
    """Cross-encoder reranking on CPU. Scores (query, "title > section: text") pairs, sets
    ``rerank_score`` and orders by it. Loads the model on first use; when loading fails it waits ten
    minutes before trying again, so a missing model does not slow every search."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model: Any = None
        self._lock = threading.Lock()
        self._load_failed_at: float | None = None
        self._warned = False

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if (
                    self._load_failed_at is not None
                    and time.monotonic() - self._load_failed_at < _RETRY_LOAD_AFTER_S
                ):
                    raise RuntimeError("the reranker model failed to load recently")
                try:
                    self._model = TextCrossEncoder(model_name=self.name)
                    self._load_failed_at = None
                except Exception:
                    self._load_failed_at = time.monotonic()
                    raise
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        try:
            model = self._get_model()
            scores = [float(s) for s in model.rerank(query, [h.chunk.embed_text for h in hits])]
            if len(scores) != len(hits):
                raise RuntimeError(f"expected {len(hits)} scores, got {len(scores)}")
        except Exception as exc:  # noqa: BLE001 - degrade to the fused order, never fail the search
            if not self._warned:
                log.warning("reranker %s failed, keeping the fused order: %s", self.name, exc)
                self._warned = True
            return list(hits[:top_n])
        rescored = [
            hit.model_copy(update={"rerank_score": score}) for hit, score in zip(hits, scores, strict=True)
        ]
        order = sorted(range(len(rescored)), key=lambda i: (-scores[i], i))
        return [rescored[i] for i in order[:top_n]]
