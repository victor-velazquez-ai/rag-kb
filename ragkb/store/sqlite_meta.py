"""SQLite metadata store: collections, documents, jobs, answers, feedback, audit and the knowledge version.

Every call opens its own short-lived connection, so FastAPI worker threads and the background
ingestion thread can share one SQLiteMeta safely. The database runs in WAL mode (readers never wait
for the writer) with a busy timeout, and the version counter is bumped inside BEGIN IMMEDIATE so two
writers can never hand out the same version.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragkb.models import Collection, Document, DocumentStatus, Feedback, Job

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    pages INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    uploaded_at TEXT NOT NULL,
    indexed_at TEXT,
    knowledge_version TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS documents_by_collection ON documents (collection_id, uploaded_at);
CREATE INDEX IF NOT EXISTS documents_by_hash ON documents (collection_id, content_hash);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_by_collection ON jobs (collection_id, created_at);
CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_by_collection ON answers (collection_id, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id TEXT NOT NULL,
    collection_id TEXT,
    rating TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_by_collection ON feedback (collection_id, created_at);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    fields TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    day TEXT NOT NULL,
    n INTEGER NOT NULL,
    value TEXT NOT NULL
);
"""

_DOCUMENT_FIELDS = set(Document.model_fields)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):  # enums
        return value.value
    return str(value)


class SQLiteMeta:
    """MetadataStore backed by one SQLite file. Safe to share across threads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._version_lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A connection inside one IMMEDIATE transaction, committed on success."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # ---------------------------------------------------------------------------------- collections

    def create_collection(self, name: str, description: str = "") -> Collection:
        collection = Collection(
            id=uuid.uuid4().hex, name=name.strip(), description=description, created_at=_now()
        )
        with self._write() as conn:
            conn.execute(
                "INSERT INTO collections (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (collection.id, collection.name, collection.description, _iso(collection.created_at)),
            )
        return collection

    def get_collection(self, collection_id: str) -> Collection | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM collections WHERE id = ?", (collection_id,)).fetchone()
        return self._collection(row) if row else None

    def list_collections(self) -> list[Collection]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM collections ORDER BY created_at, name").fetchall()
        return [self._collection(r) for r in rows]

    def delete_collection(self, collection_id: str) -> None:
        """Delete the collection and every metadata row that belongs to it."""
        with self._write() as conn:
            conn.execute("DELETE FROM feedback WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM answers WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM jobs WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

    @staticmethod
    def _collection(row: sqlite3.Row) -> Collection:
        return Collection(
            id=row["id"], name=row["name"], description=row["description"], created_at=_dt(row["created_at"])
        )

    # ------------------------------------------------------------------------------------ documents

    def create_document(self, doc: Document) -> Document:
        values = self._document_values(doc.model_dump())
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self._write() as conn:
            conn.execute(f"INSERT INTO documents ({columns}) VALUES ({marks})", tuple(values.values()))
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._document(row) if row else None

    def list_documents(self, collection_id: str) -> list[Document]:
        """Documents of one collection, newest upload first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? ORDER BY uploaded_at DESC, title",
                (collection_id,),
            ).fetchall()
        return [self._document(r) for r in rows]

    def update_document(self, doc_id: str, **fields: Any) -> Document:
        """Update some fields of a document and return it. Raises KeyError when it does not exist."""
        rejected = sorted((set(fields) - _DOCUMENT_FIELDS) | ({"id"} & set(fields)))
        if rejected:
            raise ValueError(f"Cannot update document fields: {', '.join(rejected)}.")
        if fields:
            values = self._document_values(fields)
            assignments = ", ".join(f"{column} = ?" for column in values)
            with self._write() as conn:
                cursor = conn.execute(
                    f"UPDATE documents SET {assignments} WHERE id = ?", (*values.values(), doc_id)
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document {doc_id} was not found.")
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(f"Document {doc_id} was not found.")
        return doc

    def delete_document(self, doc_id: str) -> None:
        with self._write() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def find_by_hash(self, collection_id: str, content_hash: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? AND content_hash = ? "
                "ORDER BY uploaded_at DESC LIMIT 1",
                (collection_id, content_hash),
            ).fetchone()
        return self._document(row) if row else None

    def find_by_title(self, collection_id: str, title: str) -> Document | None:
        """Title match ignoring case and surrounding spaces."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? AND lower(trim(title)) = lower(trim(?)) "
                "ORDER BY uploaded_at DESC LIMIT 1",
                (collection_id, title),
            ).fetchone()
        return self._document(row) if row else None

    @staticmethod
    def _document_values(fields: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, DocumentStatus):
                value = value.value
            elif isinstance(value, datetime):
                value = _iso(value)
            elif key == "metadata":
                value = json.dumps(value or {}, default=_json_default)
            values[key] = value
        return values

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        data = dict(row)
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        for key in ("uploaded_at", "indexed_at"):
            data[key] = _dt(data[key])
        return Document(**data)

    # ----------------------------------------------------------------------------------------- jobs

    def create_job(self, collection_id: str, doc_id: str) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            collection_id=collection_id,
            doc_id=doc_id,
            status=DocumentStatus.queued,
            created_at=_now(),
        )
        with self._write() as conn:
            conn.execute(
                "INSERT INTO jobs (id, collection_id, doc_id, status, error, created_at, finished_at) "
                "VALUES (?, ?, ?, ?, NULL, ?, NULL)",
                (job.id, job.collection_id, job.doc_id, job.status.value, _iso(job.created_at)),
            )
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def list_jobs(self, collection_id: str, limit: int = 50) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE collection_id = ? ORDER BY created_at DESC LIMIT ?",
                (collection_id, limit),
            ).fetchall()
        return [self._job(r) for r in rows]

    def update_job(
        self, job_id: str, status: DocumentStatus, error: str | None = None, *, doc_id: str | None = None
    ) -> Job:
        """Set a job's status; ready and failed also stamp finished_at. With *doc_id*, also record the
        document the job produced (a new upload has none until the pipeline returns). Raises KeyError
        when the job is missing."""
        status = DocumentStatus(status)
        finished = _iso(_now()) if status in (DocumentStatus.ready, DocumentStatus.failed) else None
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, doc_id = COALESCE(?, doc_id) WHERE id = ?",
                (status.value, error, finished, doc_id, job_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Job {job_id} was not found.")
        job = self.get_job(job_id)
        assert job is not None
        return job

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            collection_id=row["collection_id"],
            doc_id=row["doc_id"],
            status=DocumentStatus(row["status"]),
            error=row["error"],
            created_at=_dt(row["created_at"]),
            finished_at=_dt(row["finished_at"]),
        )

    # -------------------------------------------------------------------------- answers and feedback

    def save_answer(self, answer_json: str, answer_id: str, collection_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers (id, collection_id, created_at, body) VALUES (?, ?, ?, ?)",
                (answer_id, collection_id, _iso(_now()), answer_json),
            )

    def get_answer(self, answer_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT body FROM answers WHERE id = ?", (answer_id,)).fetchone()
        return row["body"] if row else None

    def add_feedback(self, feedback: Feedback) -> None:
        with self._write() as conn:
            row = conn.execute(
                "SELECT collection_id FROM answers WHERE id = ?", (feedback.answer_id,)
            ).fetchone()
            conn.execute(
                "INSERT INTO feedback (answer_id, collection_id, rating, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    feedback.answer_id,
                    row["collection_id"] if row else None,
                    feedback.rating,
                    feedback.note,
                    _iso(feedback.created_at),
                ),
            )

    def list_feedback(self, collection_id: str | None = None, limit: int = 100) -> list[Feedback]:
        with self._connect() as conn:
            if collection_id is None:
                rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feedback WHERE collection_id = ? ORDER BY id DESC LIMIT ?",
                    (collection_id, limit),
                ).fetchall()
        return [
            Feedback(
                answer_id=r["answer_id"], rating=r["rating"], note=r["note"], created_at=_dt(r["created_at"])
            )
            for r in rows
        ]

    # ---------------------------------------------------------------------------- audit and version

    def audit(self, event: str, **fields: Any) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO audit (at, event, fields) VALUES (?, ?, ?)",
                (_iso(_now()), event, json.dumps(fields, default=_json_default, sort_keys=True)),
            )

    def list_audit(self, limit: int = 100, event: str | None = None) -> list[dict[str, Any]]:
        """Newest audit rows first, as {"at", "event", **fields}."""
        with self._connect() as conn:
            if event is None:
                rows = conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit WHERE event = ? ORDER BY id DESC LIMIT ?", (event, limit)
                ).fetchall()
        return [{"at": r["at"], "event": r["event"], **json.loads(r["fields"])} for r in rows]

    def bump_version(self) -> str:
        """Next knowledge version: v{YYYYMMDD}.{n}, n counting from 1 within each UTC day."""
        today = _now().strftime("%Y%m%d")
        with self._version_lock, self._write() as conn:
            row = conn.execute("SELECT day, n FROM version WHERE id = 1").fetchone()
            n = row["n"] + 1 if row and row["day"] == today else 1
            value = f"v{today}.{n}"
            conn.execute(
                "INSERT INTO version (id, day, n, value) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET day = excluded.day, n = excluded.n, value = excluded.value",
                (today, n, value),
            )
        return value

    def current_version(self) -> str:
        """The latest knowledge version, or an empty string before anything was indexed."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM version WHERE id = 1").fetchone()
        return row["value"] if row else ""
