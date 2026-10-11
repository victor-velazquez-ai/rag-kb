"""Ingestion pipeline: file -> parse -> chunk -> embed -> store, with status, dedupe, versions, progress
and cancellation.

Identity within a collection is (source_path, content_hash). ``source_path`` is the relative path from
a folder or zip upload when the caller gives one, else the file name. Titles are for display only and
may repeat. For an upload:

- Same bytes (sha256) as a ready document anywhere in the collection, or as an upload still being
  indexed: nothing is done; the result says ``already_indexed`` and carries that document.
- Same source path, different bytes: the file was updated. The new version replaces the document in
  place (same document id), the stale tail is dropped after the new chunks are written, and the result
  names the replaced document and its previous title.
- Same bytes as a document that failed or was never indexed: that document is retried in place.
- Otherwise a new document is created. Two files with the same title and different paths are both kept.

Refused before any work, raising and leaving no row: an unsupported type, an empty file, a file over
``MAX_UPLOAD_MB``, a password-protected file, a missing collection. Accepted and then failing (no
readable text, a damaged or scanned file, a model or store failure): the document row stays with status
failed and a plain error, and its original is kept so it can be retried or dismissed; a failed
replacement keeps the previous version searchable and its previous original.

Stages are parsing, chunking, embedding, writing and done. ``progress`` is called at every stage change
and after every embedding batch with {"stage", "chunks_done", "chunks_total", "eta_seconds"}. Passages
are embedded ``EMBED_BATCH_SIZE`` at a time, longest first, so each batch pads little and the estimate
errs high. Between stages and batches the run checks for cancellation (``meta.cancel_requested(job_id)``,
or a delete of the document or its collection) and, when asked, stops, leaves the previous version
intact (or no document at all for a new upload) and raises IngestCancelled. Writing is not interrupted.

Writing order, so a failure never loses the previous version: embed every passage first, bump the
knowledge version, write all chunks of the document in one store commit, drop the stale tail, then mark
the document ready.

Locking: each document has its own lock, held while it is indexed, re-indexed or deleted, and the stores
serialize only their short writes. So deleting or re-indexing one document never waits for another
document's embedding. Deleting a document that is being indexed stops that run at its next check.

Originals are kept at ``uploads_dir/{collection_id}/{doc_id}/{filename}`` for download and re-index.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragkb.config import Settings
from ragkb.ingest.chunker import DEFAULT_MAX_TOKENS, ChunkDraft, chunk_blocks, estimate_tokens
from ragkb.ingest.parsers import (
    EmptyDocumentError,
    FileTooLargeError,
    IngestError,
    ParsedDocument,
    UnsupportedFileTypeError,
    check_not_encrypted,
    media_type_for,
    parse,
    title_from_filename,
    unsupported_message,
)
from ragkb.models import Chunk, Document, DocumentStatus
from ragkb.retrieval.base import Embedder
from ragkb.retrieval.embedder import EmbedderError
from ragkb.store.base import MetadataStore, VectorStore
from ragkb.store.lancedb_store import StoreError

log = logging.getLogger(__name__)

STAGING_DIR = ".staging"
STAGES = ("parsing", "chunking", "embedding", "writing", "done")
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

ProgressCallback = Callable[[dict[str, Any]], None]


class NotFoundError(LookupError):
    """A collection or document does not exist. ``code`` matches the API error envelope."""

    code = "not_found"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IngestCancelled(Exception):
    """An upload or re-index stopped on request. The previous version of the document, if any, is
    unchanged; a new upload leaves no document."""

    code = "cancelled"

    def __init__(self, message: str = "Indexing was cancelled. Nothing was changed.", *, doc_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.doc_id = doc_id


@dataclass(frozen=True)
class IngestResult:
    """What an upload or re-index did. ``document`` is the document as it now stands (ready or failed,
    or the existing document when ``already_indexed``)."""

    document: Document
    already_indexed: bool = False
    replaced_doc_id: str | None = None
    replaced_title: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(filename: str) -> str:
    """The last path component with characters that are unsafe on Windows or POSIX replaced."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = _UNSAFE_CHARS.sub("_", name).strip().rstrip(". ")
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    if stem.lower() in _RESERVED_NAMES:
        stem = f"_{stem}"
    stem = stem[:150]
    if not stem.strip(" _."):
        stem = "upload"
    return f"{stem}.{suffix}" if suffix else stem


def normalize_source_path(value: str | None) -> str:
    """A folder- or zip-relative path in one form: forward slashes, no drive, no leading slash, no "."
    or ".." parts. It is only an identity key; files are stored under their safe file name."""
    parts = [p.strip() for p in re.split(r"[\\/]+", (value or "").strip())]
    parts = [p for p in parts if p not in ("", ".", "..")]
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    return "/".join(parts)


def _plain_error(exc: BaseException) -> str:
    if isinstance(exc, IngestError):
        return exc.message
    if isinstance(exc, EmbedderError | StoreError):
        return str(exc)
    return "Indexing failed because of an internal error. The server log has the details."


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class _DocLocks:
    """One lock per document id, made on demand and dropped once nobody holds or waits for it."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _LockEntry] = {}

    def acquire(self, doc_id: str, *, blocking: bool = True) -> bool:
        with self._guard:
            entry = self._entries.setdefault(doc_id, _LockEntry())
            entry.users += 1
        if entry.lock.acquire(blocking):
            return True
        self._forget(doc_id)
        return False

    def release(self, doc_id: str) -> None:
        with self._guard:
            entry = self._entries[doc_id]
        entry.lock.release()
        self._forget(doc_id)

    def _forget(self, doc_id: str) -> None:
        with self._guard:
            entry = self._entries.get(doc_id)
            if entry is not None:
                entry.users -= 1
                if entry.users <= 0:
                    del self._entries[doc_id]


class _Reporter:
    """Sends progress to the caller's callback; a failing callback is logged and never stops a run."""

    def __init__(self, callback: ProgressCallback | None):
        self.callback = callback

    def __call__(self, stage: str, done: int = 0, total: int | None = None, eta: float | None = None) -> None:
        if self.callback is None:
            return
        update = {
            "stage": stage,
            "chunks_done": done,
            "chunks_total": total,
            "eta_seconds": None if eta is None else max(0, round(eta)),
        }
        try:
            self.callback(update)
        except IngestCancelled:
            raise
        except Exception:  # progress is advisory: a broken callback never stops indexing
            log.warning("a progress callback failed", exc_info=True)


@dataclass
class _Run:
    """One upload or re-index of one document, while its document lock is held."""

    event: str  # "ingest" or "reindex"
    doc_id: str
    collection_id: str
    source: Path
    media_type: str
    filename: str
    prev: Document | None  # the row before this run; None for a new upload
    explicit_title: str
    fields: dict[str, Any]  # upload fields written when the run ends (empty for a re-index)
    job_id: str | None
    report: _Reporter
    started: float
    timings: dict[str, int] = field(default_factory=dict)

    @property
    def created(self) -> bool:
        return self.prev is None

    @property
    def replacing(self) -> bool:
        """True when the document has an indexed version that must survive a failure."""
        return self.prev is not None and self.prev.chunk_count > 0


class IngestPipeline:
    def __init__(self, settings: Settings, meta: MetadataStore, store: VectorStore, embedder: Embedder):
        self.settings = settings
        self.meta = meta
        self.store = store
        self.embedder = embedder
        self._registry = threading.Lock()  # short: identity decisions and the bookkeeping below
        self._locks = _DocLocks()
        self._inflight: dict[tuple[str, str], str] = {}  # (collection id, content hash) -> doc id
        self._active: Counter[str] = Counter()  # doc id -> runs registered for it
        self._active_collection: dict[str, str] = {}  # doc id -> collection id, while active
        self._stop: set[str] = set()  # doc ids being deleted: their runs stop at the next check
        self._closing: set[str] = set()  # collection ids being deleted

    # ------------------------------------------------------------------------------------ helpers

    def original_path(self, doc: Document) -> Path:
        """Where the original upload of a document is kept."""
        return self.settings.uploads_dir / doc.collection_id / doc.id / doc.filename

    def find_duplicate(self, collection_id: str, data: bytes) -> Document | None:
        """The ready document with exactly these bytes in the collection, else the document an upload of
        these bytes is being indexed into right now, else None."""
        digest = content_hash(data)
        doc = self.meta.find_by_hash(collection_id, digest)
        if doc is not None and doc.status == DocumentStatus.ready:
            return doc
        with self._registry:
            running = self._inflight.get((collection_id, digest))
        return self.meta.get_document(running) if running is not None else None

    def _require_collection(self, collection_id: str) -> None:
        if collection_id in self._closing or self.meta.get_collection(collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} was not found.")

    def _check_upload(self, filename: str, size: int) -> None:
        """Refusals before any work: type, emptiness and size."""
        if media_type_for(filename) is None:
            raise UnsupportedFileTypeError(unsupported_message(filename))
        if size == 0:
            raise EmptyDocumentError("The file is empty.")
        if size > self.settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise FileTooLargeError(f"The file is larger than the {self.settings.MAX_UPLOAD_MB} MB limit per file.")

    def _enter(self, doc_id: str, collection_id: str) -> None:
        """Register a run for a document (call with the registry lock held)."""
        self._active[doc_id] += 1
        self._active_collection[doc_id] = collection_id

    def _leave(self, doc_id: str) -> None:
        with self._registry:
            self._active[doc_id] -= 1
            if self._active[doc_id] <= 0:
                del self._active[doc_id]
                self._active_collection.pop(doc_id, None)

    def _checkpoint(self, run: _Run) -> None:
        """Raise IngestCancelled when the run was asked to stop."""
        if run.doc_id in self._stop:
            raise IngestCancelled("The document was deleted while it was being indexed.", doc_id=run.doc_id)
        if run.collection_id in self._closing:
            raise IngestCancelled("The collection was deleted while the document was being indexed.", doc_id=run.doc_id)
        if run.job_id and self.meta.cancel_requested(run.job_id):
            raise IngestCancelled(doc_id=run.doc_id)

    def _count_tokens(self) -> Callable[[str], int]:
        return getattr(self.embedder, "count_tokens", None) or estimate_tokens

    def _max_tokens(self) -> int:
        return int(getattr(self.embedder, "max_tokens", DEFAULT_MAX_TOKENS))

    # ------------------------------------------------------------------------------------- ingest

    def ingest_bytes(
        self,
        collection_id: str,
        data: bytes,
        filename: str,
        *,
        title: str | None = None,
        source_path: str | None = None,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Ingest an upload. Raises UnsupportedFileTypeError, EmptyDocumentError, FileTooLargeError,
        EncryptedFileError or NotFoundError before any work, and IngestCancelled when cancelled; every
        later failure comes back as a document with status failed."""
        name = safe_filename(filename)
        self._check_upload(name, len(data))
        self._require_collection(collection_id)
        staging = self.settings.uploads_dir / collection_id / STAGING_DIR / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        try:
            path = staging / name
            path.write_bytes(data)
            return self.ingest_file(
                collection_id,
                path,
                title=title,
                original_filename=name,
                source_path=source_path,
                job_id=job_id,
                progress=progress,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def ingest_file(
        self,
        collection_id: str,
        path: Path,
        *,
        title: str | None = None,
        original_filename: str | None = None,
        source_path: str | None = None,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Ingest a file from disk; the original is copied into the uploads folder."""
        path = Path(path)
        filename = safe_filename(original_filename or path.name)
        media_type = media_type_for(filename)
        if media_type is None:
            raise UnsupportedFileTypeError(unsupported_message(filename))
        self._check_upload(filename, path.stat().st_size)
        self._require_collection(collection_id)
        check_not_encrypted(path, media_type)
        data = path.read_bytes()
        return self._ingest(
            collection_id,
            path,
            data,
            filename,
            media_type,
            explicit_title=(title or "").strip(),
            source_path=normalize_source_path(source_path) or filename,
            job_id=job_id,
            progress=progress,
        )

    def _ingest(
        self,
        collection_id: str,
        path: Path,
        data: bytes,
        filename: str,
        media_type: str,
        *,
        explicit_title: str,
        source_path: str,
        job_id: str | None,
        progress: ProgressCallback | None,
    ) -> IngestResult:
        started = time.monotonic()
        digest = content_hash(data)
        key = (collection_id, digest)
        fields: dict[str, Any] = {
            "filename": filename,
            "source_path": source_path,
            "media_type": media_type,
            "size_bytes": len(data),
            "content_hash": digest,
            "uploaded_at": _now(),
        }
        with self._registry:
            self._require_collection(collection_id)
            known = self.meta.find_by_hash(collection_id, digest)
            if known is not None and known.status == DocumentStatus.ready:
                return self._duplicate(known, filename)
            running = self._inflight.get(key)
            running_doc = self.meta.get_document(running) if running is not None else None
            if running_doc is not None:
                return self._duplicate(running_doc, filename)
            target = self.meta.find_by_path(collection_id, source_path)
            if target is None and known is not None:  # the same bytes never got indexed: retry them in place
                target = known
            created = target is None
            if target is None:
                target = self.meta.create_document(
                    Document(
                        id=uuid.uuid4().hex,
                        collection_id=collection_id,
                        title=explicit_title or title_from_filename(filename),
                        status=DocumentStatus.processing,
                        metadata={"title_source": "explicit" if explicit_title else "file"},
                        **fields,
                    )
                )
                self._locks.acquire(target.id, blocking=False)  # a new id: nobody else can hold it
            doc_id = target.id
            self._inflight[key] = doc_id
            self._enter(doc_id, collection_id)
        try:
            if not created:
                self._locks.acquire(doc_id)
            try:
                prev = None if created else self.meta.get_document(doc_id)
                if not created and prev is None:
                    raise IngestCancelled("The document was deleted before this upload could replace it.")
                if prev is not None and prev.status == DocumentStatus.ready and prev.content_hash == digest:
                    return self._duplicate(prev, filename)  # another upload indexed these bytes meanwhile
                run = _Run(
                    event="ingest",
                    doc_id=doc_id,
                    collection_id=collection_id,
                    source=path,
                    media_type=media_type,
                    filename=filename,
                    prev=prev,
                    explicit_title=explicit_title,
                    fields=fields,
                    job_id=job_id,
                    report=_Reporter(progress),
                    started=started,
                )
                doc = self._run(run)
            finally:
                self._locks.release(doc_id)
        finally:
            with self._registry:
                if self._inflight.get(key) == doc_id:
                    del self._inflight[key]
            self._leave(doc_id)
        replaced = (
            run.prev is not None
            and run.replacing
            and run.prev.content_hash != digest
            and doc.status == DocumentStatus.ready
        )
        return IngestResult(
            document=doc,
            replaced_doc_id=doc.id if replaced else None,
            replaced_title=run.prev.title if replaced and run.prev is not None else None,
        )

    def _duplicate(self, doc: Document, filename: str) -> IngestResult:
        self.meta.audit(
            "ingest",
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=filename,
            outcome="duplicate",
        )
        return IngestResult(document=doc, already_indexed=True)

    def reindex(
        self, doc_id: str, *, job_id: str | None = None, progress: ProgressCallback | None = None
    ) -> IngestResult:
        """Re-parse and re-embed a document from its kept original. Raises NotFoundError for an unknown
        document and IngestCancelled when cancelled (the indexed version then stays as it was)."""
        started = time.monotonic()
        with self._registry:
            doc = self.meta.get_document(doc_id)
            if doc is None:
                raise NotFoundError(f"Document {doc_id} was not found.")
            self._require_collection(doc.collection_id)
            self._enter(doc_id, doc.collection_id)
        try:
            self._locks.acquire(doc_id)
            try:
                prev = self.meta.get_document(doc_id)
                if prev is None:
                    raise NotFoundError(f"Document {doc_id} was not found.")
                run = _Run(
                    event="reindex",
                    doc_id=doc_id,
                    collection_id=prev.collection_id,
                    source=self.original_path(prev),
                    media_type=prev.media_type,
                    filename=prev.filename,
                    prev=prev,
                    explicit_title=prev.title if prev.metadata.get("title_source") != "file" else "",
                    fields={},
                    job_id=job_id,
                    report=_Reporter(progress),
                    started=started,
                )
                return IngestResult(document=self._run(run))
            finally:
                self._locks.release(doc_id)
        finally:
            self._leave(doc_id)

    # ------------------------------------------------------------------------------------ one run

    def _run(self, run: _Run) -> Document:
        """Parse, chunk, embed and write one document; the document lock is held."""
        if not run.created:
            self.meta.update_document(run.doc_id, status=DocumentStatus.processing, error=None)
        try:
            self._checkpoint(run)
            if not run.source.is_file():
                return self._fail(run, "The original file is missing, so the document cannot be re-indexed.")
            run.report("parsing")
            mark = time.monotonic()
            try:
                parsed = parse(run.source, run.media_type, skip_hidden_sheets=self.settings.SKIP_HIDDEN_SHEETS)
            except IngestError as exc:
                return self._fail(run, exc.message)
            run.timings["parse_ms"] = _ms(mark)

            self._checkpoint(run)
            run.report("chunking")
            mark = time.monotonic()
            title = run.explicit_title or parsed.title or title_from_filename(run.filename)
            drafts = chunk_blocks(
                parsed.blocks,
                title=title,
                target_tokens=self.settings.CHUNK_TOKENS,
                overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS,
                count_tokens=self._count_tokens(),
                max_tokens=self._max_tokens(),
                max_chars=self.settings.MAX_CHUNK_CHARS,
            )
            run.timings["chunk_ms"] = _ms(mark)
            if not drafts:
                return self._fail(run, "The file contains no readable text.")

            mark = time.monotonic()
            vectors = self._embed(run, drafts)
            run.timings["embed_ms"] = _ms(mark)

            self._checkpoint(run)  # the last chance to stop: the write below is not interrupted
            run.report("writing", len(drafts), len(drafts))
            mark = time.monotonic()
            version = self._write(run, drafts, vectors, title)
            run.timings["write_ms"] = _ms(mark)
        except IngestCancelled:
            self._cancelled(run)
            raise
        except Exception as exc:  # any failure leaves the previous version in place
            if not isinstance(exc, IngestError | EmbedderError | StoreError):
                log.exception("indexing failed for document %s", run.doc_id)
            return self._fail(run, _plain_error(exc))
        return self._finish(run, parsed, title, len(drafts), version)

    def _embed(self, run: _Run, drafts: list[ChunkDraft]) -> list[list[float]]:
        """Embed every passage in batches of EMBED_BATCH_SIZE, longest first; report after each batch."""
        total = len(drafts)
        order = sorted(range(total), key=lambda i: (-drafts[i].token_estimate, i))
        weights = [max(1, d.token_estimate) for d in drafts]
        total_weight = sum(weights)
        vectors: list[list[float] | None] = [None] * total
        batch_size = max(1, self.settings.EMBED_BATCH_SIZE)
        run.report("embedding", 0, total)
        began = time.monotonic()
        done = done_weight = 0
        for first in range(0, total, batch_size):
            self._checkpoint(run)
            batch = order[first : first + batch_size]
            out = self.embedder.embed([drafts[i].embed_text for i in batch])
            if len(out) != len(batch):
                raise EmbedderError(f"The embedding model returned {len(out)} vectors for {len(batch)} passages.")
            for i, vector in zip(batch, out, strict=True):
                vectors[i] = vector
            done += len(batch)
            done_weight += sum(weights[i] for i in batch)
            elapsed = time.monotonic() - began
            run.report("embedding", done, total, elapsed / done_weight * (total_weight - done_weight))
        return [v for v in vectors if v is not None]

    def _write(self, run: _Run, drafts: list[ChunkDraft], vectors: list[list[float]], title: str) -> str:
        """Bump the version, write all chunks in one commit, then drop the stale tail."""
        version = self.meta.bump_version()
        created = _now()
        chunks = [
            Chunk(
                id=f"{run.doc_id}#{ordinal}",
                doc_id=run.doc_id,
                collection_id=run.collection_id,
                ordinal=ordinal,
                text=draft.text,
                embed_text=draft.embed_text,
                title=title,
                section=draft.section,
                page=draft.page,
                page_start=draft.page_start,
                page_end=draft.page_end,
                char_start=draft.char_start,
                char_end=draft.char_end,
                token_estimate=draft.token_estimate,
                knowledge_version=version,
                created_at=created,
            )
            for ordinal, draft in enumerate(drafts)
        ]
        self.store.upsert(chunks, vectors)
        self.store.delete_doc(run.doc_id, keep_ordinals_below=len(chunks))
        return version

    def _finish(self, run: _Run, parsed: ParsedDocument, title: str, count: int, version: str) -> Document:
        if run.event == "ingest":
            try:
                self._keep_original(run.collection_id, run.doc_id, run.filename, run.source)
            except OSError as exc:
                log.error("could not keep the original of document %s: %s", run.doc_id, exc)
        metadata = dict(run.prev.metadata if run.prev is not None else {"title_source": "file"})
        if run.event == "ingest":
            metadata["title_source"] = "explicit" if run.explicit_title else "file"
        doc = self.meta.update_document(
            run.doc_id,
            **run.fields,
            title=title,
            metadata=metadata,
            status=DocumentStatus.ready,
            error=None,
            pages=parsed.pages,
            chunk_count=count,
            indexed_at=_now(),
            knowledge_version=version,
        )
        run.report("done", count, count, 0)
        elapsed_ms = _ms(run.started)
        replaced = run.event == "ingest" and run.replacing
        self.meta.audit(
            run.event,
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=doc.filename,
            media_type=doc.media_type,
            outcome="replaced" if replaced else "indexed",
            chunks=count,
            pages=parsed.pages,
            elapsed_ms=elapsed_ms,
            knowledge_version=version,
            **run.timings,
        )
        log.info(
            "%s %s (%s): %d chunks in %d ms (%s)",
            run.event,
            doc.title,
            doc.id,
            count,
            elapsed_ms,
            ", ".join(f"{k} {v}" for k, v in run.timings.items()),
        )
        return doc

    def _fail(self, run: _Run, message: str) -> Document:
        if run.replacing and run.event == "reindex":
            error = f"Re-indexing failed: {message} The previous index is still searchable."
        elif run.replacing:
            error = f"The new version could not be indexed: {message} The previous version is still searchable."
        else:
            error = message
        # A document with nothing indexed now describes the failed file, kept so it can be retried.
        fields = {} if run.replacing else run.fields
        if not run.replacing and run.event == "ingest":
            try:
                self._keep_original(run.collection_id, run.doc_id, run.filename, run.source)
            except OSError as exc:
                log.error("could not keep the original of document %s: %s", run.doc_id, exc)
        doc = self.meta.update_document(run.doc_id, **fields, status=DocumentStatus.failed, error=error)
        self.meta.audit(
            run.event,
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=doc.filename,
            outcome="failed",
            error=message,
            elapsed_ms=_ms(run.started),
        )
        log.warning("%s failed for %s (%s): %s", run.event, doc.title, doc.id, message)
        return doc

    def _cancelled(self, run: _Run) -> None:
        """Undo the visible effects of a stopped run: a new upload leaves no document; an existing
        document gets its previous status back (its chunks and original were never touched)."""
        try:
            if run.prev is None:
                self.meta.delete_document(run.doc_id)
                shutil.rmtree(self.settings.uploads_dir / run.collection_id / run.doc_id, ignore_errors=True)
            else:
                self.meta.update_document(run.doc_id, status=run.prev.status, error=run.prev.error)
        except KeyError:
            pass  # the row is already gone
        self.meta.audit(
            run.event,
            collection_id=run.collection_id,
            doc_id=run.doc_id,
            filename=run.filename,
            outcome="cancelled",
            elapsed_ms=_ms(run.started),
        )
        log.info("%s cancelled for %s (%s)", run.event, run.filename, run.doc_id)

    def _keep_original(self, collection_id: str, doc_id: str, filename: str, source: Path) -> None:
        """Copy the upload to its kept location and remove any older original of the document."""
        folder = self.settings.uploads_dir / collection_id / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / filename
        if source.resolve() != destination.resolve():
            partial = folder / f".{filename}.part"
            shutil.copyfile(source, partial)
            os.replace(partial, destination)
        for other in folder.iterdir():
            if other.is_file() and other.name != filename:
                other.unlink()

    # ------------------------------------------------------------------------------------- delete

    def delete(self, doc_id: str) -> None:
        """Delete a document's chunks, its metadata and its original file. Waits only for a run on this
        same document, which it stops at that run's next check."""
        with self._registry:
            running = doc_id in self._active
            if running:
                self._stop.add(doc_id)
        try:
            self._locks.acquire(doc_id)
            try:
                doc = self.meta.get_document(doc_id)
                if doc is None:
                    if running:
                        return  # a new upload that was stopped removed itself
                    raise NotFoundError(f"Document {doc_id} was not found.")
                removed = self.store.delete_doc(doc_id)
                self.meta.delete_document(doc_id)
                shutil.rmtree(self.settings.uploads_dir / doc.collection_id / doc.id, ignore_errors=True)
                version = self.meta.bump_version()
            finally:
                self._locks.release(doc_id)
        finally:
            if running:
                with self._registry:
                    self._stop.discard(doc_id)
        self.meta.audit(
            "delete",
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=doc.filename,
            chunks=removed,
            knowledge_version=version,
        )

    def delete_collection(self, collection_id: str) -> int:
        """Delete a collection with all its chunks, metadata and originals. Returns chunks removed.
        Runs in the collection stop at their next check; new uploads to it are refused meanwhile."""
        with self._registry:
            collection = self.meta.get_collection(collection_id)
            if collection is None or collection_id in self._closing:
                raise NotFoundError(f"Collection {collection_id} was not found.")
            self._closing.add(collection_id)
            running = sorted(d for d, c in self._active_collection.items() if c == collection_id)
        held: list[str] = []
        try:
            for doc_id in running:
                self._locks.acquire(doc_id)
                held.append(doc_id)
            removed = self.store.delete_collection(collection_id)
            self.meta.delete_collection(collection_id)
            shutil.rmtree(self.settings.uploads_dir / collection_id, ignore_errors=True)
            version = self.meta.bump_version()
        finally:
            for doc_id in held:
                self._locks.release(doc_id)
            with self._registry:
                self._closing.discard(collection_id)
        self.meta.audit(
            "delete_collection",
            collection_id=collection_id,
            name=collection.name,
            chunks=removed,
            knowledge_version=version,
        )
        return removed


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)
