"""Shared singletons, built once per app (or once per CLI run), and the operations that use them.

The routes and the command line both go through ``Deps`` so an upload, a question or a stats call
behaves the same everywhere. Each part of the system is imported defensively: when a part is not
installed (or fails to start) the container records a plain problem, the health endpoint reports
"degraded", and only the features that need that part answer 503.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, Request

from ragkb import __version__
from ragkb.config import Settings, get_settings
from ragkb.models import (
    Answer,
    AnswerStatus,
    Collection,
    Document,
    DocumentStatus,
    Job,
    ProviderStatus,
)
from ragkb.providers.base import ProviderError
from ragkb.server.errors import ApiError, plain_error

log = logging.getLogger("ragkb.server")

# Each part is optional at import time so the app starts while parts are missing.
_IMPORT_PROBLEMS: dict[str, str] = {}

try:
    from ragkb.store.sqlite_meta import SQLiteMeta
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    SQLiteMeta = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["metadata store"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.store.lancedb_store import LanceDBStore
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    LanceDBStore = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["vector store"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.retrieval.embedder import FakeEmbedder, FastEmbedEmbedder
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    FakeEmbedder = None  # type: ignore[assignment,misc]
    FastEmbedEmbedder = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["embedder"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.retrieval.reranker import LocalReranker, NoReranker
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    LocalReranker = None  # type: ignore[assignment,misc]
    NoReranker = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["reranker"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.retrieval.service import RetrievalUnavailable, Retriever
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    Retriever = None  # type: ignore[assignment,misc]

    class RetrievalUnavailable(Exception):  # type: ignore[no-redef]
        """Stand-in so the except clauses below stay valid while the retrieval part is missing."""

    _IMPORT_PROBLEMS["retriever"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.ingest.pipeline import IngestPipeline
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    IngestPipeline = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["ingestion"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.ingest.parsers import SUPPORTED_SUFFIXES, media_type_for
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    SUPPORTED_SUFFIXES = frozenset()  # type: ignore[assignment]
    media_type_for = None  # type: ignore[assignment]
    _IMPORT_PROBLEMS["parsers"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.providers.registry import ProviderRegistry
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    ProviderRegistry = None  # type: ignore[assignment,misc]
    _IMPORT_PROBLEMS["providers"] = f"{type(_exc).__name__}: {_exc}"

try:
    from ragkb.answer.compose import compose, compose_provider_error, compose_unavailable
    from ragkb.answer.compose import error_message as provider_error_message
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    compose = None  # type: ignore[assignment]
    compose_provider_error = None  # type: ignore[assignment]
    compose_unavailable = None  # type: ignore[assignment]
    provider_error_message = None  # type: ignore[assignment]
    _IMPORT_PROBLEMS["answer composer"] = f"{type(_exc).__name__}: {_exc}"


FAKE_EMBEDDER_ENV = "RAGKB_FAKE_EMBEDDER"
ASK_LIMIT_PER_MINUTE = 30
MCP_TOOLS: tuple[str, ...] = ("list_collections", "list_documents", "search_knowledge_base", "ask_knowledge_base")
PROVIDER_LABELS: dict[str, str] = {"claude_code": "Claude Code", "codex": "Codex", "extractive": "Extractive mode"}
SEARCH_UNAVAILABLE = "Search is unavailable right now."

_COMPONENT_LABELS: dict[str, str] = {
    "meta": "metadata store",
    "store": "vector store",
    "embedder": "embedder",
    "retriever": "retriever",
    "pipeline": "ingestion",
    "providers": "providers",
    "compose": "answer composer",
}


def missing_parts() -> dict[str, str]:
    """Parts that could not be imported, with the technical reason (for logs and doctor, not for users)."""
    return dict(_IMPORT_PROBLEMS)


def fake_embedder_requested() -> bool:
    """Tests and offline demos set RAGKB_FAKE_EMBEDDER=1 to use deterministic hash vectors."""
    return os.environ.get(FAKE_EMBEDDER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def provider_label(name: str | None) -> str:
    if not name:
        return "The answer provider"
    return PROVIDER_LABELS.get(name, name)


def status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


class RateLimiter:
    """Sliding-window limit per key. It is per server process: several workers each count on their own."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record one call. Returns None when allowed, or the seconds to wait when the limit is reached."""
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and now - window[0] >= self.window_s:
                window.popleft()
            if len(window) >= self.limit:
                return max(0.0, self.window_s - (now - window[0]))
            window.append(now)
            return None


@dataclass
class JobExtra:
    """What the job table does not hold: the kind of job, the file name and the dedupe outcome."""

    kind: str
    filename: str | None = None
    already_indexed: bool = False
    doc_id: str | None = None


class JobTracker:
    """In-process details for recent jobs, bounded so a long-running server does not grow without end."""

    def __init__(self, max_items: int = 5000) -> None:
        self._items: OrderedDict[str, JobExtra] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items

    def put(self, job_id: str, extra: JobExtra) -> None:
        with self._lock:
            self._items[job_id] = extra
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, job_id: str) -> JobExtra | None:
        with self._lock:
            return self._items.get(job_id)

    def set_doc(self, job_id: str, doc_id: str) -> None:
        with self._lock:
            extra = self._items.get(job_id)
            if extra is not None:
                extra.doc_id = doc_id


class JobOut(Job):
    """A job as the API returns it. ``doc_id`` is empty until a new upload has created its document."""

    kind: str = "upload"
    filename: str | None = None
    already_indexed: bool = False


class CollectionOut(Collection):
    documents: int = 0


@dataclass
class Deps:
    settings: Settings
    meta: Any = None
    embedder: Any = None
    store: Any = None
    reranker: Any = None
    retriever: Any = None
    pipeline: Any = None
    providers: Any = None
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragkb-ingest")
    )
    ask_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(ASK_LIMIT_PER_MINUTE, 60.0))
    jobs: JobTracker = field(default_factory=JobTracker)
    problems: dict[str, str] = field(default_factory=dict)
    fake_embedder: bool = False
    mcp_mounted: bool = False

    # ------------------------------------------------------------------ lifecycle

    def require(self, name: str) -> Any:
        """Return a component or raise 503 with a plain message when it is not available."""
        if name == "compose":
            if compose is None:
                raise ApiError(503, "unavailable", "Answering is not available in this installation.")
            return compose
        value = getattr(self, name, None)
        if value is None:
            label = _COMPONENT_LABELS.get(name, name)
            raise ApiError(503, "unavailable", f"The {label} is not available right now.")
        return value

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        for component in (self.store, self.meta):
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    log.warning("could not close a store cleanly", exc_info=True)

    # ------------------------------------------------------------------ small helpers

    def knowledge_version(self) -> str:
        if self.meta is None:
            return ""
        try:
            return str(self.meta.current_version())
        except Exception:
            log.warning("could not read the knowledge version", exc_info=True)
            return ""

    def audit(self, event: str, **fields: Any) -> None:
        if self.meta is None:
            return
        try:
            self.meta.audit(event, **fields)
        except Exception:
            log.warning("could not write an audit row", exc_info=True, extra={"event": event})

    def get_collection(self, collection_id: str) -> Collection:
        meta = self.require("meta")
        collection = meta.get_collection(collection_id)
        if collection is None:
            raise ApiError(404, "not_found", "Collection not found.")
        return collection

    def get_document(self, doc_id: str) -> Document:
        meta = self.require("meta")
        doc = meta.get_document(doc_id)
        if doc is None:
            raise ApiError(404, "not_found", "Document not found.")
        return doc

    def find_collection_by_name(self, name: str) -> Collection | None:
        meta = self.require("meta")
        wanted = " ".join(name.split()).casefold()
        for collection in meta.list_collections():
            if " ".join(collection.name.split()).casefold() == wanted:
                return collection
        return None

    def list_collections(self) -> list[CollectionOut]:
        meta = self.require("meta")
        out: list[CollectionOut] = []
        for collection in meta.list_collections():
            try:
                count = len(meta.list_documents(collection.id))
            except Exception:  # noqa: BLE001 - a count is optional; the list must still load
                count = 0
            out.append(CollectionOut(**collection.model_dump(), documents=count))
        return out

    def create_collection(self, name: str, description: str = "") -> Collection:
        meta = self.require("meta")
        if self.find_collection_by_name(name) is not None:
            raise ApiError(409, "conflict", "A collection with this name already exists.")
        collection = meta.create_collection(name, description)
        self.audit("collection_created", collection_id=collection.id, name=collection.name)
        return collection

    def delete_collection(self, collection_id: str) -> int:
        """Delete every document (chunks, metadata and originals), then the collection itself.

        Returns the number of documents removed.
        """
        meta = self.require("meta")
        self.get_collection(collection_id)
        documents = len(meta.list_documents(collection_id))
        remover = getattr(self.pipeline, "delete_collection", None)
        if callable(remover):
            remover(collection_id)  # removes chunks, rows and originals, bumps the version, audits
            return documents
        for doc in meta.list_documents(collection_id):
            self.delete_document(doc.id)
        if self.store is not None:
            try:
                self.store.delete_collection(collection_id)
            except Exception:
                log.warning("could not sweep the collection's chunks", exc_info=True)
        meta.delete_collection(collection_id)
        self.audit("delete_collection", collection_id=collection_id, documents=documents)
        return documents

    def delete_document(self, doc_id: str) -> None:
        doc = self.get_document(doc_id)
        if self.pipeline is not None:
            self.pipeline.delete(doc_id)  # removes chunks, row and original, bumps the version, audits
            return
        if self.store is not None:
            self.store.delete_doc(doc_id)
        self.meta.delete_document(doc_id)
        self.audit("delete", collection_id=doc.collection_id, doc_id=doc_id)

    def has_documents(self, collection_id: str) -> bool:
        if self.store is not None:
            try:
                return bool(self.store.has_documents(collection_id))
            except Exception:
                log.warning("vector store probe failed; falling back to the document table", exc_info=True)
        try:
            return any(d.status == DocumentStatus.ready for d in self.meta.list_documents(collection_id))
        except Exception:  # noqa: BLE001 - both probes failed: report no documents
            return False

    def original_path(self, doc: Document) -> Path | None:
        """Locate the kept original of a document inside the uploads folder, never outside it."""
        root = self.settings.uploads_dir.resolve()
        candidates: list[Path] = []
        finder = getattr(self.pipeline, "original_path", None)
        if callable(finder):
            try:
                candidates.append(Path(finder(doc)))
            except Exception:
                log.warning("pipeline could not locate an original", exc_info=True)
        folder = root / doc.collection_id / doc.id
        candidates.append(folder / doc.filename)
        if folder.is_dir():
            candidates.extend(sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file() and resolved.is_relative_to(root):
                return resolved
        return None

    # ------------------------------------------------------------------ jobs

    def job_view(self, job: Job) -> JobOut:
        extra = self.jobs.get(job.id)
        data = job.model_dump()
        if extra is not None:
            if extra.doc_id and not data.get("doc_id"):
                data["doc_id"] = extra.doc_id
            data.update(kind=extra.kind, filename=extra.filename, already_indexed=extra.already_indexed)
        return JobOut(**data)

    def get_job(self, job_id: str) -> JobOut:
        meta = self.require("meta")
        job = meta.get_job(job_id)
        if job is None:
            raise ApiError(404, "not_found", "Job not found.")
        return self.job_view(job)

    def list_jobs(self, collection_id: str, limit: int = 50) -> list[JobOut]:
        meta = self.require("meta")
        self.get_collection(collection_id)
        return [self.job_view(job) for job in meta.list_jobs(collection_id, limit=limit)]

    def submit_upload(self, collection_id: str, filename: str, data: bytes, title: str | None) -> JobOut:
        """Create a queued job for one file and run it in the ingestion pool; report a known hash at once."""
        meta = self.require("meta")
        self.require("pipeline")
        existing = meta.find_by_hash(collection_id, hashlib.sha256(data).hexdigest())
        if existing is not None and existing.status == DocumentStatus.ready:
            job = meta.create_job(collection_id, existing.id)
            job = meta.update_job(job.id, DocumentStatus.ready)
            self.jobs.put(job.id, JobExtra("upload", filename, already_indexed=True, doc_id=existing.id))
            # Same shape as the pipeline's own duplicate event, so the audit trail reads the same either way.
            self.audit("ingest", collection_id=collection_id, doc_id=existing.id, title=existing.title,
                       filename=filename, outcome="duplicate")
            return self.job_view(job)
        doc_id = existing.id if existing is not None else ""
        job = meta.create_job(collection_id, doc_id)
        self.jobs.put(job.id, JobExtra("upload", filename, doc_id=doc_id or None))
        self.executor.submit(self._run_upload, job.id, collection_id, filename, data, title)
        return self.job_view(job)

    def submit_reindex(self, doc_id: str) -> JobOut:
        meta = self.require("meta")
        self.require("pipeline")
        doc = self.get_document(doc_id)
        job = meta.create_job(doc.collection_id, doc.id)
        self.jobs.put(job.id, JobExtra("reindex", doc.filename, doc_id=doc.id))
        self.executor.submit(self._run_reindex, job.id, doc.id)
        return self.job_view(job)

    def _set_job(self, job_id: str, status: DocumentStatus, error: str | None = None, *,
                 doc_id: str | None = None) -> None:
        try:
            self.meta.update_job(job_id, status, error, doc_id=doc_id)
        except Exception:
            log.warning("could not update a job", exc_info=True, extra={"job_id": job_id})

    def _finish_from_document(self, job_id: str, doc: Document | None, fallback: str) -> None:
        """Close the job and record the document it produced, so the link survives a restart."""
        doc_id = doc.id if doc is not None else None
        if doc_id is not None:
            self.jobs.set_doc(job_id, doc_id)
        if doc is not None and doc.status == DocumentStatus.failed:
            self._set_job(job_id, DocumentStatus.failed, doc.error or fallback, doc_id=doc_id)
        else:
            self._set_job(job_id, DocumentStatus.ready, doc_id=doc_id)

    def _run_upload(self, job_id: str, collection_id: str, filename: str, data: bytes, title: str | None) -> None:
        fallback = "The file could not be indexed."
        self._set_job(job_id, DocumentStatus.processing)
        try:
            doc = self.pipeline.ingest_bytes(collection_id, data, filename, title=title)
        except Exception as exc:
            log.warning("ingestion failed", exc_info=True, extra={"job_id": job_id})
            self._set_job(job_id, DocumentStatus.failed, plain_error(exc, fallback))
            return
        self._finish_from_document(job_id, doc, fallback)

    def _run_reindex(self, job_id: str, doc_id: str) -> None:
        fallback = "The document could not be re-indexed."
        self._set_job(job_id, DocumentStatus.processing)
        try:
            doc = self.pipeline.reindex(doc_id)
        except Exception as exc:
            log.warning("re-index failed", exc_info=True, extra={"job_id": job_id})
            self._set_job(job_id, DocumentStatus.failed, plain_error(exc, fallback))
            return
        self._finish_from_document(job_id, doc, fallback)

    # ------------------------------------------------------------------ search and answers

    def search(self, collection_id: str, query: str, top_k: int) -> list[Any]:
        retriever = self.require("retriever")
        self.get_collection(collection_id)
        try:
            return list(retriever.search(collection_id, query, top_k))
        except RetrievalUnavailable:
            raise ApiError(503, "retrieval_unavailable", SEARCH_UNAVAILABLE) from None

    def _unavailable(self, question: str, collection_id: str, kv: str, status: AnswerStatus, message: str) -> Answer:
        return compose_unavailable(question, collection_id=collection_id, knowledge_version=kv, status=status,
                                   message=message)

    def _provider_error_answer(self, question: str, exc: ProviderError, name: str | None, collection_id: str,
                               kv: str, passages: int, model: str | None) -> Answer:
        """The answer when no provider could be selected (not installed, not signed in, over quota)."""
        if name:
            return compose_provider_error(question, exc, provider_name=name, collection_id=collection_id,
                                          knowledge_version=kv, passages_considered=passages, model=model)
        status, message = provider_error_message(exc, provider_label(None), self.settings.ANSWER_TIMEOUT_S)
        return self._unavailable(question, collection_id, kv, status, message)

    def ask(self, collection_id: str, question: str, *, top_k: int | None = None,
            provider: str | None = None, model: str | None = None) -> Answer:
        """Search, select a provider, compose, store and audit. Degraded outcomes come back as statuses."""
        meta = self.require("meta")
        self.require("compose")
        self.get_collection(collection_id)
        started = time.perf_counter()
        kv = self.knowledge_version()
        k = top_k or self.settings.TOP_K
        requested = None if provider in (None, "", "auto") else provider
        configured = (self.settings.PROVIDER or "auto").strip().lower()
        provider_name: str | None = requested or (configured if configured != "auto" else None)
        answer: Answer | None = None
        hits: list[Any] = []

        unavailable = AnswerStatus.retrieval_unavailable
        if self.retriever is None:
            answer = self._unavailable(question, collection_id, kv, unavailable, SEARCH_UNAVAILABLE)
        else:
            try:
                hits = list(self.retriever.search(collection_id, question, k))
            except RetrievalUnavailable:
                log.warning("retrieval unavailable", extra={"collection_id": collection_id})
                answer = self._unavailable(question, collection_id, kv, unavailable, SEARCH_UNAVAILABLE)
            except Exception:
                log.exception("search failed", extra={"collection_id": collection_id})
                answer = self._unavailable(question, collection_id, kv, unavailable, SEARCH_UNAVAILABLE)

        if answer is None:
            if self.providers is None:
                answer = self._unavailable(question, collection_id, kv, AnswerStatus.provider_unavailable,
                                           "No answer provider is available in this installation.")
            else:
                try:
                    chosen = self.providers.select(requested)
                    provider_name = getattr(chosen, "name", provider_name)
                    answer = compose(question, hits, chosen, self.settings, collection_id=collection_id,
                                     knowledge_version=kv, has_documents=self.has_documents(collection_id),
                                     model=model)
                except ProviderError as exc:
                    log.info("provider unavailable", extra={"provider": provider_name, "kind": exc.kind})
                    answer = self._provider_error_answer(question, exc, provider_name, collection_id, kv,
                                                         len(hits), model)
                except ValueError:
                    log.warning("unknown provider setting", extra={"provider": provider_name})
                    answer = self._unavailable(
                        question, collection_id, kv, AnswerStatus.provider_unavailable,
                        "The configured provider is not known. Set RAGKB_PROVIDER to auto, claude_code, codex "
                        "or extractive.")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            meta.save_answer(answer.model_dump_json(), answer.id, collection_id)
        except Exception:
            log.warning("could not store the answer", exc_info=True, extra={"answer_id": answer.id})
        status = status_value(answer.status)
        self.audit(
            "answer",
            collection_id=collection_id,
            answer_id=answer.id,
            question_chars=len(question),
            provider=answer.provider.name if answer.provider else provider_name,
            model=answer.provider.model if answer.provider else model,
            status=status,
            elapsed_ms=elapsed_ms,
            passages=answer.passages_considered,
            citations=len(answer.citations),
            knowledge_version=answer.knowledge_version or kv,
        )
        log.info("answer", extra={"collection_id": collection_id, "status": status, "elapsed_ms": elapsed_ms,
                                  "provider": answer.provider.name if answer.provider else provider_name})
        return answer

    def get_answer(self, answer_id: str) -> Answer:
        meta = self.require("meta")
        raw = meta.get_answer(answer_id)
        if raw is None:
            raise ApiError(404, "not_found", "Answer not found.")
        return Answer.model_validate_json(raw)

    # ------------------------------------------------------------------ providers

    def provider_statuses(self, refresh: bool = False) -> list[ProviderStatus]:
        """Provider statuses; the registry caches the CLI checks briefly, ``refresh`` checks again."""
        registry = self.require("providers")
        try:
            return list(registry.statuses(refresh=refresh))
        except TypeError:
            return list(registry.statuses())

    # ------------------------------------------------------------------ stats and health

    def collection_stats(self, collection_id: str) -> dict[str, Any]:
        meta = self.require("meta")
        collection = self.get_collection(collection_id)
        docs: list[Document] = meta.list_documents(collection_id)
        by_status = Counter(status_value(d.status) for d in docs)
        by_type = Counter(d.media_type for d in docs)
        chunks: int | None = None
        if self.store is not None:
            try:
                chunks = int(self.store.count(collection_id))
            except Exception:
                log.warning("could not count chunks", exc_info=True)
        if chunks is None:
            chunks = sum(d.chunk_count for d in docs)
        indexed = [d.indexed_at for d in docs if d.indexed_at is not None]
        return {
            "collection_id": collection.id,
            "name": collection.name,
            "documents": len(docs),
            "chunks": chunks,
            "pages": sum(d.pages or 0 for d in docs),
            "size_bytes": sum(d.size_bytes for d in docs),
            "by_status": dict(by_status),
            "by_media_type": dict(by_type),
            "last_indexed_at": max(indexed).isoformat() if indexed else None,
            "knowledge_version": self.knowledge_version(),
        }

    def health(self) -> dict[str, Any]:
        store_state = {"metadata": "unavailable", "vector": "unavailable"}
        counts: dict[str, int | None] = {"collections": None, "documents": None, "chunks": None}
        if self.meta is not None:
            try:
                collections = self.meta.list_collections()
                counts["collections"] = len(collections)
                counts["documents"] = sum(len(self.meta.list_documents(c.id)) for c in collections)
                store_state["metadata"] = "ok"
            except Exception:
                log.warning("metadata store health check failed", exc_info=True)
        if self.store is not None:
            try:
                counts["chunks"] = int(self.store.count())
                store_state["vector"] = "ok"
            except Exception:
                log.warning("vector store health check failed", exc_info=True)
        ok = all(v == "ok" for v in store_state.values()) and not self.problems
        return {
            "status": "ok" if ok else "degraded",
            "version": __version__,
            "knowledge_version": self.knowledge_version(),
            "counts": counts,
            "store": store_state,
            "problems": [f"{name}: {reason}" for name, reason in sorted(self.problems.items())],
            "data_dir": str(self.settings.DATA_DIR.resolve()),
            "embed_model": getattr(self.embedder, "name", None) or self.settings.EMBED_MODEL,
            "embed_dims": getattr(self.embedder, "dims", None),
            "fake_embedder": self.fake_embedder,
            "reranker": self.settings.RERANKER,
            "rerank_model": self.settings.RERANK_MODEL if self.settings.RERANKER == "local" else None,
            "top_k": self.settings.TOP_K,
            "min_score": self.settings.MIN_SCORE,
            "max_context_chars": self.settings.MAX_CONTEXT_CHARS,
            "max_upload_mb": self.settings.MAX_UPLOAD_MB,
            "supported_types": supported_types(),
            "provider_setting": self.settings.PROVIDER,
            "claude_model": self.settings.CLAUDE_MODEL,
            "codex_model": self.settings.CODEX_MODEL or None,
            "ask_rate_limit": {"per_minute": self.ask_limiter.limit, "scope": "per collection, per server process"},
            "api": {"public": self.settings.API_PUBLIC},
            "mcp": {
                "enabled": self.settings.MCP_ENABLED,
                "mounted": self.mcp_mounted,
                "path": "/mcp",
                "public": self.settings.MCP_PUBLIC,
                "tools": list(MCP_TOOLS) if self.mcp_mounted else [],
            },
        }


def _try_build(problems: dict[str, str], label: str, factory: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        # Our own errors carry a plain sentence (an unknown embedding model, a dimension mismatch):
        # report it as the reason. Anything else is unexpected: keep the trace in the log.
        own = type(exc).__module__.startswith("ragkb.")
        reason = plain_error(exc, "could not start")
        log.warning("could not start a component", exc_info=not own, extra={"component": label, "reason": reason})
        problems[label] = reason
        return None


def build_deps(settings: Settings | None = None, *, fake_embedder: bool | None = None) -> Deps:
    """Build every component once. A part that is missing or fails is recorded, never fatal."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    use_fake = fake_embedder_requested() if fake_embedder is None else fake_embedder
    problems: dict[str, str] = {}
    for label, reason in _IMPORT_PROBLEMS.items():
        log.warning("part not available", extra={"component": label, "reason": reason})
        problems[label] = "not installed"

    meta = _try_build(problems, "metadata store", SQLiteMeta, settings.meta_path) if SQLiteMeta else None

    embedder = None
    if use_fake and FakeEmbedder is not None:
        embedder = _try_build(problems, "embedder", FakeEmbedder)
    elif FastEmbedEmbedder is not None:
        embedder = _try_build(problems, "embedder", FastEmbedEmbedder, settings.EMBED_MODEL)

    store = None
    if LanceDBStore is not None and embedder is not None:
        store = _try_build(problems, "vector store", LanceDBStore, settings.lancedb_dir, embedder.dims)

    reranker = None
    if settings.RERANKER == "local" and LocalReranker is not None:
        reranker = _try_build(problems, "reranker", LocalReranker, settings.RERANK_MODEL)
    if reranker is None and NoReranker is not None:
        reranker = NoReranker()

    retriever = None
    if Retriever is not None and store is not None and embedder is not None:
        retriever = _try_build(problems, "retriever", Retriever, store, embedder, reranker=reranker,
                               min_score=settings.MIN_SCORE)

    pipeline = None
    if IngestPipeline is not None and meta is not None and store is not None and embedder is not None:
        pipeline = _try_build(problems, "ingestion", IngestPipeline, settings, meta, store, embedder)

    providers = _try_build(problems, "providers", ProviderRegistry, settings) if ProviderRegistry else None

    return Deps(
        settings=settings,
        meta=meta,
        embedder=embedder,
        store=store,
        reranker=reranker,
        retriever=retriever,
        pipeline=pipeline,
        providers=providers,
        problems=problems,
        fake_embedder=bool(use_fake and embedder is not None),
    )


def get_deps(request: Request) -> Deps:
    """FastAPI dependency: the container built by create_app."""
    return request.app.state.deps


DepsDep = Annotated[Deps, Depends(get_deps)]


def supported_media_type(filename: str) -> str | None:
    if media_type_for is None:
        return None
    try:
        return media_type_for(filename)
    except Exception:  # noqa: BLE001 - an odd file name is simply unsupported
        return None


def supported_types() -> list[str]:
    return sorted(SUPPORTED_SUFFIXES)
