"""Ingestion pipeline: file -> parse -> chunk -> embed -> store, with status, dedupe and versions.

Identity rules for an upload into a collection:

- Same bytes (sha256) as a ready document: nothing is done and that document is returned, so the
  caller can report "already indexed" (``find_duplicate`` answers the same question up front).
- Same title as an existing document: the upload replaces it under the same document id. The title is
  the one given by the caller, else the one found in the file (first heading, document properties),
  else one made from the file name. Matching ignores case.
- Otherwise a new document is created.

Writing order, so a failure never loses the previous version: embed every chunk first, bump the
knowledge version, upsert all chunks in one commit, drop the stale tail (ordinals past the new count),
then mark the document ready. On any failure the document is marked failed with a plain error, the
previous chunks stay searchable and, for a replacement, the previous original file is kept.

Originals are kept at ``uploads_dir/{collection_id}/{doc_id}/{filename}`` for download and re-index.
All mutations run under one lock, so ingestion, re-index and delete never interleave.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragkb.config import Settings
from ragkb.ingest.chunker import chunk_blocks
from ragkb.ingest.parsers import (
    EmptyDocumentError,
    IngestError,
    ParsedDocument,
    UnsupportedFileTypeError,
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
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class NotFoundError(LookupError):
    """A collection or document does not exist. ``code`` matches the API error envelope."""

    code = "not_found"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


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


def _plain_error(exc: BaseException) -> str:
    if isinstance(exc, IngestError):
        return exc.message
    if isinstance(exc, EmbedderError | StoreError):
        return str(exc)
    return "Indexing failed because of an internal error. The server log has the details."


class IngestPipeline:
    def __init__(self, settings: Settings, meta: MetadataStore, store: VectorStore, embedder: Embedder):
        self.settings = settings
        self.meta = meta
        self.store = store
        self.embedder = embedder
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------------------ helpers

    def original_path(self, doc: Document) -> Path:
        """Where the original upload of a document is kept."""
        return self.settings.uploads_dir / doc.collection_id / doc.id / doc.filename

    def find_duplicate(self, collection_id: str, data: bytes) -> Document | None:
        """The ready document with exactly these bytes in the collection, if any."""
        doc = self.meta.find_by_hash(collection_id, content_hash(data))
        return doc if doc is not None and doc.status == DocumentStatus.ready else None

    def _require_collection(self, collection_id: str) -> None:
        if self.meta.get_collection(collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} was not found.")

    # ------------------------------------------------------------------------------------- ingest

    def ingest_bytes(
        self, collection_id: str, data: bytes, filename: str, *, title: str | None = None
    ) -> Document:
        """Ingest an upload. Raises UnsupportedFileTypeError, EmptyDocumentError (empty bytes) or
        NotFoundError up front; later failures come back as a document with status failed."""
        name = safe_filename(filename)
        if media_type_for(name) is None:
            raise UnsupportedFileTypeError(unsupported_message(name))
        if not data:
            raise EmptyDocumentError("The file is empty.")
        self._require_collection(collection_id)
        staging = self.settings.uploads_dir / collection_id / STAGING_DIR / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        try:
            path = staging / name
            path.write_bytes(data)
            return self.ingest_file(collection_id, path, title=title, original_filename=name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def ingest_file(
        self,
        collection_id: str,
        path: Path,
        *,
        title: str | None = None,
        original_filename: str | None = None,
    ) -> Document:
        """Ingest a file from disk; the original is copied into the uploads folder."""
        path = Path(path)
        filename = safe_filename(original_filename or path.name)
        media_type = media_type_for(filename)
        if media_type is None:
            raise UnsupportedFileTypeError(unsupported_message(filename))
        self._require_collection(collection_id)
        data = path.read_bytes()
        if not data:
            raise EmptyDocumentError("The file is empty.")
        with self._lock:
            return self._ingest(collection_id, path, data, filename, media_type, title)

    def _ingest(
        self,
        collection_id: str,
        path: Path,
        data: bytes,
        filename: str,
        media_type: str,
        explicit_title: str | None,
    ) -> Document:
        started = time.monotonic()
        digest = content_hash(data)
        duplicate = self.meta.find_by_hash(collection_id, digest)
        if duplicate is not None and duplicate.status == DocumentStatus.ready:
            self.meta.audit(
                "ingest",
                collection_id=collection_id,
                doc_id=duplicate.id,
                title=duplicate.title,
                filename=filename,
                outcome="duplicate",
            )
            return duplicate

        parsed: ParsedDocument | None = None
        parse_error: str | None = None
        try:
            parsed = parse(path, media_type)
        except IngestError as exc:
            parse_error = exc.message

        title = (
            (explicit_title or "").strip()
            or (parsed.title if parsed else "")
            or title_from_filename(filename)
        )
        target = duplicate if duplicate is not None else self.meta.find_by_title(collection_id, title)
        replacing = target is not None and target.chunk_count > 0
        upload: dict[str, Any] = {
            "title": title,
            "filename": filename,
            "media_type": media_type,
            "size_bytes": len(data),
            "content_hash": digest,
            "uploaded_at": _now(),
        }
        if target is None:
            doc = self.meta.create_document(
                Document(
                    id=uuid.uuid4().hex,
                    collection_id=collection_id,
                    status=DocumentStatus.processing,
                    **upload,
                )
            )
            self._keep_original(doc.collection_id, doc.id, filename, path)
        elif not replacing:
            doc = self.meta.update_document(
                target.id, status=DocumentStatus.processing, error=None, pages=None, **upload
            )
            self._keep_original(doc.collection_id, doc.id, filename, path)
        else:
            # the row keeps describing the indexed version until the new one is fully written
            doc = self.meta.update_document(target.id, status=DocumentStatus.processing, error=None)

        if parsed is None:
            return self._fail(doc, parse_error or "The file could not be read.", replacing, "ingest", started)
        return self._index(
            doc, parsed, path, replacing=replacing, fields=upload, event="ingest", started=started
        )

    def reindex(self, doc_id: str) -> Document:
        """Re-parse and re-embed a document from its kept original."""
        with self._lock:
            doc = self.meta.get_document(doc_id)
            if doc is None:
                raise NotFoundError(f"Document {doc_id} was not found.")
            started = time.monotonic()
            replacing = doc.chunk_count > 0
            doc = self.meta.update_document(doc.id, status=DocumentStatus.processing, error=None)
            path = self.original_path(doc)
            if not path.is_file():
                return self._fail(
                    doc,
                    "The original file is missing, so the document cannot be re-indexed.",
                    replacing,
                    "reindex",
                    started,
                )
            try:
                parsed = parse(path, doc.media_type)
            except IngestError as exc:
                return self._fail(doc, exc.message, replacing, "reindex", started)
            return self._index(
                doc, parsed, path, replacing=replacing, fields={}, event="reindex", started=started
            )

    def _index(
        self,
        doc: Document,
        parsed: ParsedDocument,
        source: Path,
        *,
        replacing: bool,
        fields: dict[str, Any],
        event: str,
        started: float,
    ) -> Document:
        title = fields.get("title", doc.title)
        try:
            drafts = chunk_blocks(
                parsed.blocks,
                title=title,
                target_tokens=self.settings.CHUNK_TOKENS,
                overlap_tokens=self.settings.CHUNK_OVERLAP_TOKENS,
            )
            if not drafts:
                raise EmptyDocumentError("The file contains no readable text.")
            vectors = self.embedder.embed([d.embed_text for d in drafts])
            if len(vectors) != len(drafts):
                raise EmbedderError(
                    f"The embedding model returned {len(vectors)} vectors for {len(drafts)} passages."
                )
            version = self.meta.bump_version()
            created = _now()
            chunks = [
                Chunk(
                    id=f"{doc.id}#{ordinal}",
                    doc_id=doc.id,
                    collection_id=doc.collection_id,
                    ordinal=ordinal,
                    text=draft.text,
                    embed_text=draft.embed_text,
                    title=title,
                    section=draft.section,
                    page=draft.page,
                    char_start=draft.char_start,
                    char_end=draft.char_end,
                    token_estimate=draft.token_estimate,
                    knowledge_version=version,
                    created_at=created,
                )
                for ordinal, draft in enumerate(drafts)
            ]
            self.store.upsert(chunks, vectors)
            self.store.delete_doc(doc.id, keep_ordinals_below=len(chunks))
        except Exception as exc:  # any failure leaves the previous version in place
            if not isinstance(exc, IngestError | EmbedderError | StoreError):
                log.exception("indexing failed for document %s", doc.id)
            return self._fail(doc, _plain_error(exc), replacing, event, started)

        filename = fields.get("filename", doc.filename)
        if replacing:
            try:
                self._keep_original(doc.collection_id, doc.id, filename, source)
            except OSError as exc:
                log.error("could not keep the original of document %s: %s", doc.id, exc)
        doc = self.meta.update_document(
            doc.id,
            **fields,
            status=DocumentStatus.ready,
            error=None,
            pages=parsed.pages,
            chunk_count=len(chunks),
            indexed_at=_now(),
            knowledge_version=version,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self.meta.audit(
            event,
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=doc.filename,
            media_type=doc.media_type,
            outcome="replaced" if replacing and event == "ingest" else "indexed",
            chunks=len(chunks),
            pages=parsed.pages,
            elapsed_ms=elapsed_ms,
            knowledge_version=version,
        )
        log.info("%s %s (%s): %d chunks in %d ms", event, doc.title, doc.id, len(chunks), elapsed_ms)
        return doc

    def _fail(self, doc: Document, message: str, replacing: bool, event: str, started: float) -> Document:
        if replacing and event == "reindex":
            error = f"Re-indexing failed: {message} The previous index is still searchable."
        elif replacing:
            error = (
                f"The new version could not be indexed: {message} The previous version is still searchable."
            )
        else:
            error = message
        doc = self.meta.update_document(doc.id, status=DocumentStatus.failed, error=error)
        self.meta.audit(
            event,
            collection_id=doc.collection_id,
            doc_id=doc.id,
            title=doc.title,
            filename=doc.filename,
            outcome="failed",
            error=message,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        log.warning("%s failed for %s (%s): %s", event, doc.title, doc.id, message)
        return doc

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
        """Delete a document's chunks, its metadata and its original file."""
        with self._lock:
            doc = self.meta.get_document(doc_id)
            if doc is None:
                raise NotFoundError(f"Document {doc_id} was not found.")
            removed = self.store.delete_doc(doc_id)
            self.meta.delete_document(doc_id)
            shutil.rmtree(self.settings.uploads_dir / doc.collection_id / doc.id, ignore_errors=True)
            version = self.meta.bump_version()
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
        """Delete a collection with all its chunks, metadata and originals. Returns chunks removed."""
        with self._lock:
            collection = self.meta.get_collection(collection_id)
            if collection is None:
                raise NotFoundError(f"Collection {collection_id} was not found.")
            removed = self.store.delete_collection(collection_id)
            self.meta.delete_collection(collection_id)
            shutil.rmtree(self.settings.uploads_dir / collection_id, ignore_errors=True)
            version = self.meta.bump_version()
            self.meta.audit(
                "delete_collection",
                collection_id=collection_id,
                name=collection.name,
                chunks=removed,
                knowledge_version=version,
            )
            return removed
