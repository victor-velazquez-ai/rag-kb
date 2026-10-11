"""The document count rule shared by the API, the health check and the MCP tools.

The headline number is always the documents that are ready (searchable). Everything else (failed,
queued, processing, cancelled) is secondary and reported by status next to it.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

IN_PROGRESS = ("queued", "processing")


def status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def document_counts(meta: Any, collection_id: str) -> dict[str, int]:
    """Documents of one collection by status value, for example ``{"ready": 10, "failed": 1}``.

    Uses the store's ``count_documents``; a store without it (a test double) is counted row by row.
    """
    counter = getattr(meta, "count_documents", None)
    if callable(counter):
        raw = dict(counter(collection_id))
    else:
        raw = dict(Counter(status_value(d.status) for d in meta.list_documents(collection_id)))
    return {status_value(k): int(v) for k, v in raw.items() if int(v)}


def headline(by_status: dict[str, int]) -> dict[str, Any]:
    """``documents`` (ready), ``documents_total`` and ``documents_by_status`` for one set of counts."""
    return {
        "documents": int(by_status.get("ready", 0)),
        "documents_total": int(sum(by_status.values())),
        "documents_by_status": dict(sorted(by_status.items())),
    }
