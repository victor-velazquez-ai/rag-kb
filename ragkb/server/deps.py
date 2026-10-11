"""Shared singletons, built once per app (or once per CLI run), and the operations that use them.

The routes and the command line both go through ``Deps`` so an upload, a question or a stats call
behaves the same everywhere. Each part of the system is imported defensively: when a part is not
installed (or fails to start) the container records a plain problem, the health endpoint reports
"degraded", and only the features that need that part answer 503.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, Request
from pydantic import Field

from ragkb import __version__
from ragkb.config import Settings, get_settings
from ragkb.models import (
    Answer,
    AnswerStatus,
    Collection,
    Document,
    DocumentStatus,
    Feedback,
    Job,
    ProviderStatus,
)
from ragkb.providers.base import ProviderError
from ragkb.server.counts import IN_PROGRESS, document_counts, headline, status_value
from ragkb.server.errors import ApiError, plain_error
from ragkb.server.jobs import IngestScheduler, QueuedJob, SchedulerStopped
from ragkb.server.text import clean_text

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
    from ragkb.ingest.pipeline import IngestCancelled
except Exception:  # noqa: BLE001 - an ingestion part without cancellation still runs

    class IngestCancelled(Exception):  # type: ignore[no-redef]
        """Stand-in so the except clause in the job runner stays valid without the ingestion part."""

try:
    from ragkb.ingest.parsers import (
        SUPPORTED_SUFFIXES,
        EncryptedFileError,
        check_data_not_encrypted,
        media_type_for,
    )
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    SUPPORTED_SUFFIXES = frozenset()  # type: ignore[assignment]
    media_type_for = None  # type: ignore[assignment]
    check_data_not_encrypted = None  # type: ignore[assignment]

    class EncryptedFileError(Exception):  # type: ignore[no-redef]
        """Stand-in so the except clause in encrypted_reason stays valid without the parsers part."""

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

# Ingestion jobs
OPEN_STATUSES = frozenset(IN_PROGRESS)
SPOOL_DIR = ".incoming"  # under the uploads folder: uploads waiting for a worker
PROGRESS_FIELDS = ("stage", "chunks_done", "chunks_total", "eta_seconds")
PROGRESS_INTERVAL_S = 0.5
DELETE_WAIT_S = 15.0
SUMMARY_WINDOW = timedelta(hours=24)
JOB_SCAN_LIMIT = 100_000
FEEDBACK_SCAN_LIMIT = 10_000
INTERRUPTED_MESSAGE = "Indexing stopped because the server was shut down. Try again."
SHUTDOWN_MESSAGE = "The server stopped before this file was indexed. Upload it again."
SPOOL_LOST_MESSAGE = "The uploaded file was lost before it could be indexed. Upload it again."
SPOOL_WRITE_MESSAGE = "The file could not be saved for indexing. Check the free disk space and try again."
DELETE_PENDING_MESSAGE = "The delete is still running and will finish in the background."

# Health fields shown to a caller that is not on this machine: nothing that names a local folder,
# a user or a program path.
PUBLIC_HEALTH_KEYS = ("status", "version", "knowledge_version", "counts", "store", "problems", "api", "mcp")

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


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Whether ``fn`` takes the keyword argument ``name`` (or any keyword)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def compose_search_options(retriever: Any, settings: Settings) -> dict[str, Any]:
    """Keyword arguments for ``Retriever.search`` when the passages go to the composer: the context
    budget, so the retriever returns every passage that fits plus neighbours of the best ones. A
    retriever without that keyword (a test double) is called with the plain arguments."""
    search = getattr(retriever, "search", None)
    if callable(search) and accepts_keyword(search, "budget_chars"):
        return {"budget_chars": settings.MAX_CONTEXT_CHARS}
    return {}


class JobInputMissing(RuntimeError):
    """A queued upload's spooled bytes are gone. The message is safe to show."""


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
    """What the job table does not hold: the kind of job and the folder path of an upload."""

    kind: str
    source_path: str = ""


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


class JobOut(Job):
    """A job as the API returns it. ``doc_id`` is empty until a new upload has created its document.

    ``kind`` is upload, reindex or retry; ``source_path`` is the file's path in a folder or zip upload.
    Both are known to the server process that ran the job (after a restart a job reads as an upload).
    """

    kind: str = "upload"
    source_path: str = ""


class CollectionOut(Collection):
    """``documents`` counts the ready (searchable) documents; the rest are in ``documents_by_status``."""

    documents: int = 0
    documents_total: int = 0
    documents_by_status: dict[str, int] = Field(default_factory=dict)


def _log_background_failure(future: Future[Any]) -> None:
    error = future.exception()
    if error is not None:
        log.warning("a delete that finished in the background failed",
                    exc_info=(type(error), error, error.__traceback__))


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
    scheduler: IngestScheduler | None = None
    # deletes run here, apart from the ingestion workers, and the request waits up to delete_wait_s
    maintenance: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragkb-delete")
    )
    delete_wait_s: float = DELETE_WAIT_S
    ask_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(ASK_LIMIT_PER_MINUTE, 60.0))
    jobs: JobTracker = field(default_factory=JobTracker)
    problems: dict[str, str] = field(default_factory=dict)
    fake_embedder: bool = False
    mcp_mounted: bool = False

    def __post_init__(self) -> None:
        if self.scheduler is None:
            self.scheduler = IngestScheduler(self.settings.INGEST_WORKERS, self._run_job)

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
        """Stop the workers (jobs that never started end as failed), then close the stores."""
        if self.scheduler is not None:
            for item in self.scheduler.shutdown():
                self._abandon(item, SHUTDOWN_MESSAGE)
        self.maintenance.shutdown(wait=False)
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
        wanted = clean_text(name).casefold()
        for collection in meta.list_collections():
            if clean_text(collection.name).casefold() == wanted:
                return collection
        return None

    def list_collections(self) -> list[CollectionOut]:
        meta = self.require("meta")
        out: list[CollectionOut] = []
        for collection in meta.list_collections():
            try:
                counts = headline(document_counts(meta, collection.id))
            except Exception:  # a count is optional; the list must still load
                log.warning("could not count a collection's documents", exc_info=True)
                counts = headline({})
            out.append(CollectionOut(**collection.model_dump(), **counts))
        return out

    def create_collection(self, name: str, description: str = "") -> Collection:
        meta = self.require("meta")
        name, description = clean_text(name), clean_text(description)
        if not name:
            raise ApiError(400, "invalid_request", "The collection name has no visible characters.")
        if self.find_collection_by_name(name) is not None:
            raise ApiError(409, "conflict", "A collection with this name already exists.")
        collection = meta.create_collection(name, description)
        self.audit("collection_created", collection_id=collection.id, name=collection.name)
        return collection

    def delete_collection(self, collection_id: str) -> tuple[int, bool]:
        """Delete every document (chunks, metadata and originals), then the collection itself.

        Waiting and running jobs of the collection are cancelled first. Returns the number of
        documents and whether the delete finished within ``delete_wait_s`` (it continues otherwise).
        """
        meta = self.require("meta")
        self.get_collection(collection_id)
        documents = int(sum(document_counts(meta, collection_id).values()))
        for job_id in self.scheduler.active(collection_id=collection_id):
            self._cancel_quietly(job_id)
        return documents, self._bounded(self._delete_collection_now, collection_id, documents)

    def _delete_collection_now(self, collection_id: str, documents: int) -> None:
        remover = getattr(self.pipeline, "delete_collection", None)
        if callable(remover):
            try:
                remover(collection_id)  # removes chunks, rows and originals, bumps the version, audits
            except Exception as exc:
                if getattr(exc, "code", None) == "not_found":  # deleted meanwhile by another request
                    raise ApiError(404, "not_found", "Collection not found.") from None
                raise
            return
        for doc in self.meta.list_documents(collection_id):
            self._delete_document_now(doc)
        if self.store is not None:
            try:
                self.store.delete_collection(collection_id)
            except Exception:
                log.warning("could not sweep the collection's chunks", exc_info=True)
        self.meta.delete_collection(collection_id)
        self.audit("delete_collection", collection_id=collection_id, documents=documents)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document now, apart from the ingestion workers.

        Its waiting or running jobs are cancelled first. Returns True when the delete finished within
        ``delete_wait_s``, False when it is still running (it finishes in the background).
        """
        doc = self.get_document(doc_id)
        for job_id in self.scheduler.active(doc_id=doc_id):
            self._cancel_quietly(job_id)
        return self._bounded(self._delete_document_now, doc)

    def _delete_document_now(self, doc: Document) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.delete(doc.id)  # removes chunks, row and original, bumps the version, audits
            except Exception as exc:
                if getattr(exc, "code", None) == "not_found":
                    raise ApiError(404, "not_found", "Document not found.") from None
                raise
            return
        if self.store is not None:
            self.store.delete_doc(doc.id)
        self.meta.delete_document(doc.id)
        self.audit("delete", collection_id=doc.collection_id, doc_id=doc.id)

    def _bounded(self, fn: Callable[..., Any], *args: Any) -> bool:
        """Run ``fn`` off the ingestion workers and wait up to ``delete_wait_s``: True when it finished."""
        future = self.maintenance.submit(fn, *args)
        try:
            future.result(timeout=self.delete_wait_s)
        except TimeoutError:
            if future.done():
                raise
            future.add_done_callback(_log_background_failure)
            log.warning("a delete is taking longer than expected and continues in the background")
            return False
        return True

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

    # ------------------------------------------------------------------ jobs: reading

    def job_view(self, job: Job) -> JobOut:
        extra = self.jobs.get(job.id)
        data = job.model_dump()
        if extra is not None:
            data.update(kind=extra.kind, source_path=extra.source_path)
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

    def jobs_summary(self, collection_id: str, since: datetime | None = None) -> dict[str, Any]:
        """Counts of the collection's jobs created since ``since`` (default: the last 24 hours).

        Each job counts once: ``replaced`` and ``duplicates`` are finished jobs that replaced an older
        version or matched an indexed file, and are not also counted in ``ready``.
        """
        meta = self.require("meta")
        self.get_collection(collection_id)
        since = since or utcnow() - SUMMARY_WINDOW
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        counts = dict.fromkeys(("queued", "processing", "ready", "failed", "cancelled", "replaced", "duplicates"), 0)
        for job in meta.list_jobs(collection_id, limit=JOB_SCAN_LIMIT):
            created = job.created_at if job.created_at.tzinfo else job.created_at.replace(tzinfo=UTC)
            if created < since:
                continue
            status = status_value(job.status)
            if status == "ready" and job.already_indexed:
                counts["duplicates"] += 1
            elif status == "ready" and job.replaced_doc_id:
                counts["replaced"] += 1
            elif status in counts:
                counts[status] += 1
        return {"collection_id": collection_id, "since": since.isoformat(), **counts, "total": sum(counts.values())}

    # ------------------------------------------------------------------ jobs: submitting

    def submit_upload(self, collection_id: str, filename: str, data: bytes, *, title: str | None = None,
                      source_path: str | None = None) -> JobOut:
        """Queue one accepted file. Bytes already indexed in the collection are reported at once."""
        self.require("meta")
        pipeline = self.require("pipeline")
        duplicate = pipeline.find_duplicate(collection_id, data)
        if duplicate is not None and status_value(duplicate.status) == DocumentStatus.ready.value:
            return self._duplicate_job(collection_id, filename, len(data), duplicate, source_path)
        job = self.meta.create_job(collection_id, "", filename=filename, bytes_total=len(data))
        self.jobs.put(job.id, JobExtra("upload", source_path or ""))
        try:
            spool = self._spool(job.id, data)
        except OSError:
            log.warning("could not spool an upload", exc_info=True, extra={"job_id": job.id})
            self._update_job(job.id, DocumentStatus.failed, SPOOL_WRITE_MESSAGE, stage="failed")
            return self.get_job(job.id)
        item = QueuedJob(job.id, "upload", collection_id, len(data), filename=filename, title=title,
                         source_path=source_path, spool=spool)
        return self._enqueue(item, job)

    def _duplicate_job(self, collection_id: str, filename: str, size: int, duplicate: Document,
                       source_path: str | None) -> JobOut:
        job = self.meta.create_job(collection_id, duplicate.id, filename=filename, bytes_total=size)
        self.jobs.put(job.id, JobExtra("upload", source_path or ""))
        self._update_job(job.id, DocumentStatus.ready, doc_id=duplicate.id, stage="done", already_indexed=True,
                         chunks_done=duplicate.chunk_count, chunks_total=duplicate.chunk_count)
        # Same shape as the pipeline's own duplicate event, so the audit trail reads the same either way.
        self.audit("ingest", collection_id=collection_id, doc_id=duplicate.id, title=duplicate.title,
                   filename=filename, outcome="duplicate")
        return self.get_job(job.id)

    def submit_reindex(self, doc_id: str, *, retry: bool = False) -> JobOut:
        """Queue a re-index from the kept original. A retry is for a failed document and clears its error."""
        meta = self.require("meta")
        self.require("pipeline")
        doc = self.get_document(doc_id)
        if retry and status_value(doc.status) != DocumentStatus.failed.value:
            raise ApiError(409, "conflict", "Only a document that failed can be retried. Use re-index for the others.")
        if self.scheduler.active(doc_id=doc.id):
            raise ApiError(409, "conflict", "This document is already waiting or being indexed.")
        kind = "retry" if retry else "reindex"
        job = meta.create_job(doc.collection_id, doc.id, filename=doc.filename, bytes_total=doc.size_bytes)
        self.jobs.put(job.id, JobExtra(kind, doc.source_path))
        item = QueuedJob(job.id, kind, doc.collection_id, doc.size_bytes, doc_id=doc.id, filename=doc.filename)
        if retry:
            item.previous_error = doc.error
            meta.update_document(doc.id, status=DocumentStatus.queued, error=None)
        return self._enqueue(item, job)

    def _enqueue(self, item: QueuedJob, job: Job) -> JobOut:
        try:
            self.scheduler.submit(item)
        except SchedulerStopped:
            self._abandon(item, SHUTDOWN_MESSAGE)
            return self.get_job(item.job_id)
        return self.job_view(job)

    def _spool_dir(self) -> Path:
        return self.settings.uploads_dir / SPOOL_DIR

    def _spool(self, job_id: str, data: bytes) -> Path:
        folder = self._spool_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / job_id
        path.write_bytes(data)
        return path

    @staticmethod
    def _drop_spool(item: QueuedJob) -> None:
        if item.spool is not None:
            try:
                item.spool.unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove a spooled upload", exc_info=True, extra={"job_id": item.job_id})

    # ------------------------------------------------------------------ jobs: running

    def _update_job(self, job_id: str, status: DocumentStatus | None = None, error: str | None = None, *,
                    doc_id: str | None = None, **progress: Any) -> None:
        try:
            self.meta.update_job(job_id, status, error, doc_id=doc_id, **progress)
        except Exception:
            log.warning("could not update a job", exc_info=True, extra={"job_id": job_id})

    def _cancel_requested(self, job_id: str) -> bool:
        try:
            return bool(self.meta.cancel_requested(job_id))
        except Exception:
            log.warning("could not read a job's cancel flag", exc_info=True, extra={"job_id": job_id})
            return False

    def _progress_writer(self, job_id: str) -> Callable[[Mapping[str, Any]], None]:
        """The pipeline's progress callback: writes stage, chunk counts and ETA to the job, at most every
        half second within one stage (a new stage, or the last chunk, is always written)."""
        last: dict[str, Any] = {"stage": None, "at": 0.0}

        def write(info: Mapping[str, Any]) -> None:
            now = time.monotonic()
            stage = info.get("stage")
            done, total = info.get("chunks_done"), info.get("chunks_total")
            finished = total is not None and done is not None and done >= total
            if stage == last["stage"] and now - last["at"] < PROGRESS_INTERVAL_S and not finished:
                return
            last.update(stage=stage, at=now)
            self._update_job(job_id, **{key: info[key] for key in PROGRESS_FIELDS if key in info})

        return write

    def _read_spool(self, item: QueuedJob) -> bytes:
        try:
            if item.spool is None:
                raise OSError("no spooled upload")
            return item.spool.read_bytes()
        except OSError:
            raise JobInputMissing(SPOOL_LOST_MESSAGE) from None

    def _run_job(self, item: QueuedJob) -> None:
        """One job on a worker: queued becomes processing, then ready, failed or cancelled."""
        try:
            if self._cancel_requested(item.job_id):
                self._end_cancelled(item)
                return
            self._update_job(item.job_id, DocumentStatus.processing, stage="parsing")
            progress = self._progress_writer(item.job_id)
            fallback = ("The file could not be indexed." if item.kind == "upload"
                        else "The document could not be re-indexed.")
            try:
                if item.kind == "upload":
                    result = self.pipeline.ingest_bytes(
                        item.collection_id, self._read_spool(item), item.filename, title=item.title,
                        source_path=item.source_path, job_id=item.job_id, progress=progress)
                else:
                    result = self.pipeline.reindex(item.doc_id, job_id=item.job_id, progress=progress)
            except IngestCancelled:
                self._end_cancelled(item)
            except Exception as exc:
                log.warning("ingestion failed", exc_info=True, extra={"job_id": item.job_id, "kind": item.kind})
                self._update_job(item.job_id, DocumentStatus.failed, plain_error(exc, fallback), stage="failed")
            else:
                self._finish(item, result, fallback)
        finally:
            self._drop_spool(item)

    def _finish(self, item: QueuedJob, result: Any, fallback: str) -> None:
        """Close the job from the pipeline's result and record the document, so the link survives a restart."""
        doc: Document | None = result.document
        doc_id = doc.id if doc is not None else None
        outcome = {
            "already_indexed": bool(result.already_indexed),
            "replaced_doc_id": result.replaced_doc_id,
            "replaced_title": result.replaced_title,
        }
        status = status_value(doc.status) if doc is not None else DocumentStatus.failed.value
        if status == DocumentStatus.failed.value:
            error = (doc.error if doc is not None else None) or fallback
            self._update_job(item.job_id, DocumentStatus.failed, error, doc_id=doc_id, stage="failed", **outcome)
        elif status == DocumentStatus.cancelled.value:
            self._update_job(item.job_id, DocumentStatus.cancelled, None, doc_id=doc_id, stage="cancelled", **outcome)
        else:
            chunks = doc.chunk_count if doc is not None else 0
            self._update_job(item.job_id, DocumentStatus.ready, None, doc_id=doc_id, stage="done",
                             chunks_done=chunks, chunks_total=chunks, eta_seconds=0, **outcome)

    def _end_cancelled(self, item: QueuedJob) -> None:
        self._update_job(item.job_id, DocumentStatus.cancelled, None, stage="cancelled")
        if item.kind == "retry":
            self._restore_retry(item)

    def _abandon(self, item: QueuedJob, message: str) -> None:
        """A job that will never run: it ends as failed with a plain reason."""
        self._update_job(item.job_id, DocumentStatus.failed, message, stage="failed")
        self._drop_spool(item)
        if item.kind == "retry":
            self._restore_retry(item)

    def _restore_retry(self, item: QueuedJob) -> None:
        """A retry that did not run puts the document back as it was: failed, with its error."""
        try:
            doc = self.meta.get_document(item.doc_id)
            if doc is not None and status_value(doc.status) in OPEN_STATUSES:
                self.meta.update_document(doc.id, status=DocumentStatus.failed, error=item.previous_error)
        except Exception:
            log.warning("could not restore a document after a retry stopped", exc_info=True,
                        extra={"doc_id": item.doc_id})

    def cancel_job(self, job_id: str) -> JobOut:
        """Cancel a waiting or running job. A finished job comes back unchanged.

        A waiting job ends as cancelled at once. A running job is flagged; the pipeline stops at its
        next checkpoint, keeps any previous version, and the worker records ``cancelled``.
        """
        meta = self.require("meta")
        job = meta.get_job(job_id)
        if job is None:
            raise ApiError(404, "not_found", "Job not found.")
        if status_value(job.status) not in OPEN_STATUSES:
            return self.job_view(job)
        try:
            meta.request_cancel(job_id)
        except Exception:
            log.warning("could not flag a job for cancellation", exc_info=True, extra={"job_id": job_id})
        where, item = self.scheduler.cancel(job_id)
        if where == "queued" and item is not None:
            self._end_cancelled(item)
            self._drop_spool(item)
        elif where == "idle":
            # Not waiting or running here: left by a server that stopped, so nothing else will close it.
            current = meta.get_job(job_id)
            if current is not None and status_value(current.status) in OPEN_STATUSES:
                self._update_job(job_id, DocumentStatus.cancelled, None, stage="cancelled")
        self.audit("job_cancelled", collection_id=job.collection_id, job_id=job_id, state=where)
        return self.get_job(job_id)

    def _cancel_quietly(self, job_id: str) -> None:
        try:
            self.cancel_job(job_id)
        except Exception:
            log.warning("could not cancel a job before a delete", exc_info=True, extra={"job_id": job_id})

    def recover_interrupted_jobs(self) -> int:
        """At startup: jobs that a stopped server left queued or processing end as failed with a plain
        reason, so do documents they left half-way, and spooled uploads nobody will read are removed."""
        if self.meta is None:
            return 0
        recovered = 0
        try:
            collections = self.meta.list_collections()
        except Exception:
            log.warning("could not look for interrupted jobs", exc_info=True)
            return 0
        for collection in collections:
            try:
                jobs = self.meta.list_jobs(collection.id, limit=JOB_SCAN_LIMIT)
            except Exception:
                log.warning("could not list a collection's jobs", exc_info=True)
                continue
            for job in jobs:
                if status_value(job.status) not in OPEN_STATUSES or self.scheduler.state(job.id) != "idle":
                    continue
                self._update_job(job.id, DocumentStatus.failed, INTERRUPTED_MESSAGE, stage="failed")
                self._fail_stuck_document(job.doc_id)
                recovered += 1
        active = set(self.scheduler.active())
        folder = self._spool_dir()
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file() and path.name not in active:
                    try:
                        path.unlink()
                    except OSError:
                        log.warning("could not remove a stale spooled upload", exc_info=True)
        if recovered:
            log.warning("interrupted jobs closed as failed", extra={"jobs": recovered})
        return recovered

    def _fail_stuck_document(self, doc_id: str) -> None:
        if not doc_id:
            return
        try:
            doc = self.meta.get_document(doc_id)
            if doc is None or status_value(doc.status) not in OPEN_STATUSES:
                return
            message = INTERRUPTED_MESSAGE
            if doc.chunk_count:
                message += " The previous version is still searchable."
            self.meta.update_document(doc.id, status=DocumentStatus.failed, error=message)
        except Exception:
            log.warning("could not close a document left half-way", exc_info=True, extra={"doc_id": doc_id})

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
                hits = list(self.retriever.search(collection_id, question, k,
                                                  **compose_search_options(self.retriever, self.settings)))
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
                    # A model the provider does not know is refused here, before any call, with a plain
                    # message that names it and the models to choose from (status provider_unavailable).
                    chosen = self.providers.select(requested, model=model)
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

    # ------------------------------------------------------------------ feedback

    def add_feedback(self, answer_id: str, rating: str, note: str = "") -> Feedback:
        meta = self.require("meta")
        answer = self.get_answer(answer_id)
        item = Feedback(id=uuid.uuid4().hex, answer_id=answer.id, collection_id=answer.collection_id,
                        question=answer.question, rating=rating, note=note.strip(), created_at=utcnow())
        meta.add_feedback(item)
        self.audit("feedback", collection_id=answer.collection_id, answer_id=answer.id, rating=rating,
                   has_note=bool(item.note))
        return item

    def list_feedback(self, collection_id: str, limit: int = 100) -> list[Feedback]:
        """A collection's votes, newest first."""
        meta = self.require("meta")
        self.get_collection(collection_id)
        return list(meta.list_feedback(collection_id, limit=limit))

    def answer_feedback(self, answer_id: str) -> list[Feedback]:
        """The votes on one answer, newest first."""
        meta = self.require("meta")
        answer = self.get_answer(answer_id)
        return [f for f in meta.list_feedback(answer.collection_id, limit=FEEDBACK_SCAN_LIMIT)
                if f.answer_id == answer.id]

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
        """Counts for one collection. The headline ``documents`` is the ready documents, and pages, size
        and media types describe those; every status is in ``documents_by_status``."""
        meta = self.require("meta")
        collection = self.get_collection(collection_id)
        docs: list[Document] = meta.list_documents(collection_id)
        by_status = dict(Counter(status_value(d.status) for d in docs))
        ready = [d for d in docs if status_value(d.status) == DocumentStatus.ready.value]
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
            **headline(by_status),
            "documents_ready": len(ready),
            "by_status": dict(sorted(by_status.items())),  # the same as documents_by_status, for older clients
            "chunks": chunks,
            "pages": sum(d.pages or 0 for d in ready),
            "size_bytes": sum(d.size_bytes for d in ready),
            "by_media_type": dict(Counter(d.media_type for d in ready)),
            "last_indexed_at": max(indexed).isoformat() if indexed else None,
            "knowledge_version": self.knowledge_version(),
        }

    def health(self, *, full: bool = True) -> dict[str, Any]:
        """Status, version, counts and store health. ``full=False`` (a caller not on this machine) leaves
        out everything local: the data folder, models, settings and supported types."""
        store_state = {"metadata": "unavailable", "vector": "unavailable"}
        counts: dict[str, int | None] = dict.fromkeys(
            ("collections", "documents", "documents_ready", "documents_failed", "documents_processing",
             "documents_total", "chunks"))
        if self.meta is not None:
            try:
                collections = self.meta.list_collections()
                by_status: Counter[str] = Counter()
                for collection in collections:
                    by_status.update(document_counts(self.meta, collection.id))
                ready = by_status.get(DocumentStatus.ready.value, 0)
                counts.update(
                    collections=len(collections),
                    documents=ready,
                    documents_ready=ready,
                    documents_failed=by_status.get(DocumentStatus.failed.value, 0),
                    documents_processing=sum(by_status.get(s, 0) for s in IN_PROGRESS),
                    documents_total=sum(by_status.values()),
                )
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
        payload: dict[str, Any] = {
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
        if not full:
            return {**{key: payload[key] for key in PUBLIC_HEALTH_KEYS}, "view": "public"}
        return {**payload, "view": "full"}


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
        # MIN_SCORE, CANDIDATES and NEIGHBOUR_EXPANSION come from the settings.
        retriever = _try_build(problems, "retriever", Retriever.from_settings, store, embedder, settings,
                               reranker=reranker)

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


def encrypted_reason(filename: str, data: bytes) -> str | None:
    """The plain reason a password-protected PDF, Word or Excel upload is refused, or None."""
    if check_data_not_encrypted is None:
        return None
    try:
        check_data_not_encrypted(data, supported_media_type(filename))
    except EncryptedFileError as exc:
        return str(getattr(exc, "message", "") or exc)
    return None
