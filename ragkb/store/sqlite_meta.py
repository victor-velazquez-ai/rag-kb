"""SQLite metadata store: collections, documents, jobs, answers, feedback, audit and the knowledge version.

Every call opens its own short-lived connection, so FastAPI worker threads and the background
ingestion threads can share one SQLiteMeta safely. The database runs in WAL mode (readers never wait
for the writer) with a busy timeout, and the version counter is bumped inside BEGIN IMMEDIATE so two
writers can never hand out the same version.

Columns added after the first release (``_ADDED_COLUMNS``) are created by a small migration on open, so
a data folder from an older version keeps working.
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
from typing import Any, get_args

from ragkb.models import Collection, Document, DocumentStatus, Feedback, Job, JobStage

SCHEMA_VERSION = 2

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

# table -> column -> definition; added by the migration when missing
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "documents": {"source_path": "TEXT NOT NULL DEFAULT ''"},
    "jobs": {
        "filename": "TEXT NOT NULL DEFAULT ''",
        "stage": "TEXT NOT NULL DEFAULT 'queued'",
        "chunks_done": "INTEGER NOT NULL DEFAULT 0",
        "chunks_total": "INTEGER",
        "bytes_total": "INTEGER",
        "eta_seconds": "INTEGER",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
        "already_indexed": "INTEGER NOT NULL DEFAULT 0",
        "replaced_doc_id": "TEXT",
        "replaced_title": "TEXT",
    },
    "feedback": {"uid": "TEXT NOT NULL DEFAULT ''", "question": "TEXT NOT NULL DEFAULT ''"},
}
_INDEX_ON_ADDED_COLUMNS = "CREATE INDEX IF NOT EXISTS documents_by_path ON documents (collection_id, source_path)"
# Rows written before a column existed get the value it implies: a document's source path is its file
# name (so uploading the same file again replaces it), and a job's stage follows from its status.
_BACKFILL = {
    ("documents", "source_path"): "UPDATE documents SET source_path = filename",
    ("jobs", "stage"): (
        "UPDATE jobs SET stage = CASE status WHEN 'ready' THEN 'done' WHEN 'failed' THEN 'failed' "
        "WHEN 'cancelled' THEN 'cancelled' ELSE 'queued' END"
    ),
}

_DOCUMENT_FIELDS = set(Document.model_fields)
_JOB_STAGES = frozenset(get_args(JobStage))
_TERMINAL = {DocumentStatus.ready: "done", DocumentStatus.failed: "failed", DocumentStatus.cancelled: "cancelled"}
# keyword fields of update_job besides status, error and doc_id
_JOB_PROGRESS_FIELDS = frozenset(
    {
        "stage",
        "chunks_done",
        "chunks_total",
        "eta_seconds",
        "already_indexed",
        "replaced_doc_id",
        "replaced_title",
        "filename",
        "bytes_total",
    }
)


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


def _answer_question(body: str) -> str:
    """The question of a saved answer, or an empty string when its JSON has none."""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    question = data.get("question") if isinstance(data, dict) else None
    return question if isinstance(question, str) else ""


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
            self._migrate(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add the columns of later versions to tables created by an older one, in one transaction so
        two processes opening the same folder cannot both add a column."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table, columns in _ADDED_COLUMNS.items():
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column, definition in columns.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                        if (table, column) in _BACKFILL:
                            conn.execute(_BACKFILL[(table, column)])
            conn.execute(_INDEX_ON_ADDED_COLUMNS)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

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

    def count_documents(self, collection_id: str) -> dict[str, int]:
        """Documents of one collection by status value, for example {"ready": 10, "failed": 1}."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents WHERE collection_id = ? GROUP BY status",
                (collection_id,),
            ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def find_by_hash(self, collection_id: str, content_hash: str) -> Document | None:
        """The document with these bytes in the collection; a ready one wins over any other."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? AND content_hash = ? "
                "ORDER BY (status = 'ready') DESC, uploaded_at DESC LIMIT 1",
                (collection_id, content_hash),
            ).fetchone()
        return self._document(row) if row else None

    def find_by_path(self, collection_id: str, source_path: str) -> Document | None:
        """The document uploaded from this source path (exact match) in the collection."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? AND source_path = ? "
                "ORDER BY uploaded_at DESC LIMIT 1",
                (collection_id, source_path),
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

    def create_job(
        self, collection_id: str, doc_id: str, *, filename: str = "", bytes_total: int | None = None
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            collection_id=collection_id,
            doc_id=doc_id,
            status=DocumentStatus.queued,
            filename=filename,
            bytes_total=bytes_total,
            created_at=_now(),
        )
        with self._write() as conn:
            conn.execute(
                "INSERT INTO jobs (id, collection_id, doc_id, status, error, created_at, finished_at, filename, "
                "stage, bytes_total) VALUES (?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)",
                (
                    job.id,
                    job.collection_id,
                    job.doc_id,
                    job.status.value,
                    _iso(job.created_at),
                    job.filename,
                    job.stage,
                    job.bytes_total,
                ),
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
        self,
        job_id: str,
        status: DocumentStatus | None = None,
        error: str | None = None,
        *,
        doc_id: str | None = None,
        **progress: Any,
    ) -> Job:
        """Update a job and return it. Raises KeyError when the job is missing and ValueError for an
        unknown progress field or stage.

        With a status, the status and the error are set; ready, failed and cancelled also stamp
        finished_at and, unless a stage is given, set the stage to done, failed or cancelled. Without a
        status (None) only progress is written: the status stays, and the error changes only when one
        is given. With *doc_id*, also record the document the job produced (a new upload has none until
        the pipeline returns). Progress fields: stage, chunks_done, chunks_total, eta_seconds,
        already_indexed, replaced_doc_id, replaced_title, filename, bytes_total."""
        unknown = sorted(set(progress) - _JOB_PROGRESS_FIELDS)
        if unknown:
            raise ValueError(f"Cannot update job fields: {', '.join(unknown)}.")
        if "stage" in progress and progress["stage"] not in _JOB_STAGES:
            raise ValueError(f"Unknown job stage: {progress['stage']}.")
        assignments: list[str] = []
        values: list[Any] = []
        if status is not None:
            status = DocumentStatus(status)
            assignments += ["status = ?", "error = ?", "finished_at = ?"]
            values += [status.value, error, _iso(_now()) if status in _TERMINAL else None]
            if status in _TERMINAL and "stage" not in progress:
                progress["stage"] = _TERMINAL[status]
        elif error is not None:
            assignments.append("error = ?")
            values.append(error)
        if doc_id is not None:
            assignments.append("doc_id = ?")
            values.append(doc_id)
        for key, value in progress.items():
            if key == "eta_seconds" and value is not None:
                value = max(0, round(value))
            elif key == "already_indexed":
                value = int(bool(value))
            assignments.append(f"{key} = ?")
            values.append(value)
        if assignments:
            with self._write() as conn:
                cursor = conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", (*values, job_id))
                if cursor.rowcount == 0:
                    raise KeyError(f"Job {job_id} was not found.")
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} was not found.")
        return job

    def request_cancel(self, job_id: str) -> Job | None:
        """Ask a running job to stop; the pipeline stops at its next check. A finished job is left as
        it is. Returns the job, or None when it does not exist."""
        with self._write() as conn:
            conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND finished_at IS NULL", (job_id,))
        return self.get_job(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            collection_id=row["collection_id"],
            doc_id=row["doc_id"],
            status=DocumentStatus(row["status"]),
            error=row["error"],
            filename=row["filename"],
            stage=row["stage"],
            chunks_done=row["chunks_done"],
            chunks_total=row["chunks_total"],
            bytes_total=row["bytes_total"],
            eta_seconds=row["eta_seconds"],
            cancel_requested=bool(row["cancel_requested"]),
            already_indexed=bool(row["already_indexed"]),
            replaced_doc_id=row["replaced_doc_id"],
            replaced_title=row["replaced_title"],
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
        """Store a rating. The collection and the question come from the saved answer when it exists
        (else from the feedback itself); an empty id gets a new one."""
        with self._write() as conn:
            row = conn.execute(
                "SELECT collection_id, body FROM answers WHERE id = ?", (feedback.answer_id,)
            ).fetchone()
            collection_id = (row["collection_id"] if row else None) or feedback.collection_id or None
            question = (_answer_question(row["body"]) if row is not None else "") or feedback.question
            conn.execute(
                "INSERT INTO feedback (uid, answer_id, collection_id, question, rating, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback.id or uuid.uuid4().hex,
                    feedback.answer_id,
                    collection_id,
                    question,
                    feedback.rating,
                    feedback.note,
                    _iso(feedback.created_at),
                ),
            )

    def list_feedback(self, collection_id: str | None = None, limit: int = 100) -> list[Feedback]:
        """Newest feedback first, of one collection or of all."""
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
                id=r["uid"] or str(r["id"]),
                answer_id=r["answer_id"],
                collection_id=r["collection_id"] or "",
                question=r["question"] or "",
                rating=r["rating"],
                note=r["note"],
                created_at=_dt(r["created_at"]),
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
