"""HTTP API tests. Offline: fake embedder, extractive provider, a temporary data folder.

Two kinds of test live here:

- Server behaviour (lanes, labels, progress, cancel, deletes, uploads, zip archives, counts, health,
  names, errors, feedback, retry, the job summary) runs against ``MemoryMeta`` and ``FakePipeline``,
  in-memory doubles that follow the store and pipeline contracts (ragkb/store/base.py and the
  pinned ingestion contract), so these tests do not depend on the real stores.
- End-to-end tests on the real stores and pipeline skip until the ingestion and store part with
  progress, cancellation and source paths is installed, and the answer tests skip without the
  providers and answer part.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import threading
import time
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ragkb.config import Settings
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
from ragkb.server.app import create_app
from ragkb.server.deps import INTERRUPTED_MESSAGE, Deps, IngestCancelled, RateLimiter, build_deps
from tests.conftest import build_pdf
from tests.test_parsers import encrypted_pdf

CORE_PARTS = (
    "ragkb.store.sqlite_meta",
    "ragkb.store.lancedb_store",
    "ragkb.retrieval.embedder",
    "ragkb.retrieval.service",
    "ragkb.ingest.parsers",
    "ragkb.ingest.pipeline",
)
ANSWER_PARTS = ("ragkb.providers.registry", "ragkb.answer.compose")
PART_A_REASON = "needs the ingestion and store part with progress, cancellation and source paths"

MB = 1024 * 1024
MARKDOWN = (
    b"# Leave policy\n\n"
    b"## Annual leave\n\n"
    b"Every employee gets 25 days of annual leave per calendar year. Up to five unused days carry over "
    b"to the next year and expire at the end of March.\n\n"
    b"## Sick leave\n\n"
    b"Sick leave longer than three consecutive days needs a doctor's note sent to the people team.\n"
)
TEXT = (
    b"Expense claims\n\n"
    b"Expense claims must be filed within 30 days of the purchase. Receipts are required for any amount "
    b"above 20 euros. Claims are paid with the next monthly salary.\n"
)
ANSWER_FIELDS = set(Answer.model_fields)
# The API answers only loopback hosts unless RAGKB_API_PUBLIC is set.
LOCAL_BASE = "http://127.0.0.1:8765"
REMOTE_BASE = "http://kb.example.com"
RLO = chr(0x202E)
FINISHED = ("ready", "failed", "cancelled")


def _part_a_present() -> bool:
    """The ingestion and store part this server is built against: cancellation, counts, source paths."""
    try:
        pipeline_mod = importlib.import_module("ragkb.ingest.pipeline")
        meta_mod = importlib.import_module("ragkb.store.sqlite_meta")
    except Exception:  # noqa: BLE001 - any import failure means the part is absent
        return False
    store = meta_mod.SQLiteMeta
    needed = ("request_cancel", "cancel_requested", "count_documents", "find_by_path")
    return hasattr(pipeline_mod, "IngestCancelled") and all(hasattr(store, name) for name in needed)


PART_A = _part_a_present()


def _require(*modules: str) -> None:
    for name in modules:
        pytest.importorskip(name)


def now() -> datetime:
    return datetime.now(UTC)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- in-memory doubles


class MemoryMeta:
    """The MetadataStore contract in memory. Thread-safe; every read returns a copy."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.collections: dict[str, Collection] = {}
        self.documents: dict[str, Document] = {}
        self.jobs: dict[str, Job] = {}
        self.order: dict[str, int] = {}
        self.answers: dict[str, tuple[str, str]] = {}
        self.feedback: list[Feedback] = []
        self.audits: list[tuple[str, dict[str, Any]]] = []
        self.version = 0

    # collections
    def create_collection(self, name: str, description: str = "") -> Collection:
        collection = Collection(id=uuid.uuid4().hex, name=name, description=description, created_at=now())
        with self.lock:
            self.collections[collection.id] = collection
        return collection

    def get_collection(self, collection_id: str) -> Collection | None:
        return self.collections.get(collection_id)

    def list_collections(self) -> list[Collection]:
        with self.lock:
            return list(self.collections.values())

    def delete_collection(self, collection_id: str) -> None:
        with self.lock:
            self.collections.pop(collection_id, None)
            for store in (self.documents, self.jobs):
                for key in [k for k, v in store.items() if v.collection_id == collection_id]:
                    store.pop(key)
            self.feedback = [f for f in self.feedback if f.collection_id != collection_id]

    # documents
    def create_document(self, doc: Document) -> Document:
        with self.lock:
            self.documents[doc.id] = doc
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        with self.lock:
            doc = self.documents.get(doc_id)
            return doc.model_copy() if doc else None

    def list_documents(self, collection_id: str) -> list[Document]:
        with self.lock:
            docs = [d for d in self.documents.values() if d.collection_id == collection_id]
        return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)

    def count_documents(self, collection_id: str) -> dict[str, int]:
        return dict(Counter(d.status.value for d in self.list_documents(collection_id)))

    def update_document(self, doc_id: str, **fields: Any) -> Document:
        with self.lock:
            doc = Document.model_validate({**self.documents[doc_id].model_dump(), **fields})
            self.documents[doc_id] = doc
            return doc

    def delete_document(self, doc_id: str) -> None:
        with self.lock:
            self.documents.pop(doc_id, None)

    def _find(self, collection_id: str, test: Callable[[Document], bool]) -> Document | None:
        matches = [d for d in self.list_documents(collection_id) if test(d)]
        return matches[0] if matches else None

    def find_by_hash(self, collection_id: str, content_hash: str) -> Document | None:
        return self._find(collection_id, lambda d: d.content_hash == content_hash)

    def find_by_path(self, collection_id: str, source_path: str) -> Document | None:
        return self._find(collection_id, lambda d: (d.source_path or d.filename) == source_path)

    def find_by_title(self, collection_id: str, title: str) -> Document | None:
        return self._find(collection_id, lambda d: d.title.casefold() == title.casefold())

    # jobs
    def create_job(self, collection_id: str, doc_id: str, *, filename: str = "",
                   bytes_total: int | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, collection_id=collection_id, doc_id=doc_id, status=DocumentStatus.queued,
                  filename=filename, bytes_total=bytes_total, created_at=now())
        with self.lock:
            self.jobs[job.id] = job
            self.order[job.id] = len(self.order)
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy() if job else None

    def list_jobs(self, collection_id: str, limit: int = 50) -> list[Job]:
        with self.lock:
            jobs = [j for j in self.jobs.values() if j.collection_id == collection_id]
            jobs.sort(key=lambda j: (j.created_at, self.order.get(j.id, 0)), reverse=True)
            return jobs[:limit]

    def update_job(self, job_id: str, status: DocumentStatus | None = None, error: str | None = None, *,
                   doc_id: str | None = None, **progress: Any) -> Job:
        with self.lock:
            update = dict(progress)
            if status is not None:
                update.update(status=DocumentStatus(status), error=error)
                if status in (DocumentStatus.ready, DocumentStatus.failed, DocumentStatus.cancelled):
                    update["finished_at"] = now()
            if doc_id is not None:
                update["doc_id"] = doc_id
            job = Job.model_validate({**self.jobs[job_id].model_dump(), **update})
            self.jobs[job_id] = job
            return job

    def request_cancel(self, job_id: str) -> Job | None:
        with self.lock:
            if job_id not in self.jobs:
                return None
            return self.update_job(job_id, cancel_requested=True)

    def cancel_requested(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            return bool(job and job.cancel_requested)

    # answers and feedback
    def save_answer(self, answer_json: str, answer_id: str, collection_id: str) -> None:
        self.answers[answer_id] = (collection_id, answer_json)

    def get_answer(self, answer_id: str) -> str | None:
        stored = self.answers.get(answer_id)
        return stored[1] if stored else None

    def add_feedback(self, feedback: Feedback) -> None:
        with self.lock:
            self.feedback.append(feedback)

    def list_feedback(self, collection_id: str | None = None, limit: int = 100) -> list[Feedback]:
        with self.lock:
            rows = [f for f in reversed(self.feedback) if collection_id is None or f.collection_id == collection_id]
        return rows[:limit]

    # audit and version
    def audit(self, event: str, **fields: Any) -> None:
        self.audits.append((event, fields))

    def bump_version(self) -> str:
        with self.lock:
            self.version += 1
            return self.current_version()

    def current_version(self) -> str:
        return f"v20261010.{self.version}" if self.version else ""


@dataclass
class FakeResult:
    document: Document | None
    already_indexed: bool = False
    replaced_doc_id: str | None = None
    replaced_title: str | None = None


class FakePipeline:
    """The pinned ingestion contract over MemoryMeta.

    ``hold(name)`` keeps that file (by file name) inside the pipeline until the returned event is set;
    while held it honours a cancel request like the real pipeline. ``steps[name]`` lists progress
    reports sent before it waits. Same bytes as a ready document come back as already indexed; the
    same source path replaces the document; a file name starting with ``broken`` fails.
    """

    def __init__(self, meta: MemoryMeta) -> None:
        self.meta = meta
        self.gates: dict[str, threading.Event] = {}
        self.started: dict[str, threading.Event] = {}
        self.steps: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []
        self.delete_gate: threading.Event | None = None

    def hold(self, name: str) -> threading.Event:
        self.started[name] = threading.Event()
        self.gates[name] = threading.Event()
        return self.gates[name]

    def release_all(self) -> None:
        for gate in self.gates.values():
            gate.set()
        if self.delete_gate is not None:
            self.delete_gate.set()

    def _wait(self, name: str, job_id: str | None, progress: Callable[[dict[str, Any]], None] | None) -> None:
        for step in self.steps.get(name, []):
            if progress is not None:
                progress(step)
        if name in self.started:
            self.started[name].set()
        gate = self.gates.get(name)
        while gate is not None and not gate.wait(0.01):
            if job_id and self.meta.cancel_requested(job_id):
                raise IngestCancelled("cancelled")

    def find_duplicate(self, collection_id: str, data: bytes) -> Document | None:
        doc = self.meta.find_by_hash(collection_id, sha(data))
        live = (DocumentStatus.ready, DocumentStatus.queued, DocumentStatus.processing)
        return doc if doc is not None and doc.status in live else None

    def ingest_bytes(self, collection_id: str, data: bytes, filename: str, *, title: str | None = None,
                     source_path: str | None = None, job_id: str | None = None,
                     progress: Callable[[dict[str, Any]], None] | None = None) -> FakeResult:
        self.calls.append({"filename": filename, "title": title, "source_path": source_path, "job_id": job_id})
        self._wait(filename, job_id, progress)
        existing = self.meta.find_by_hash(collection_id, sha(data))
        if existing is not None and existing.status == DocumentStatus.ready:
            return FakeResult(existing, already_indexed=True)
        path = source_path or filename
        fields = {"title": title or filename.rsplit(".", 1)[0], "filename": filename, "source_path": path,
                  "media_type": "text/plain", "size_bytes": len(data), "content_hash": sha(data),
                  "uploaded_at": now()}
        if filename.startswith("broken"):
            doc = Document(id=uuid.uuid4().hex, collection_id=collection_id, status=DocumentStatus.failed,
                           error="The file could not be read.", **fields)
            return FakeResult(self.meta.create_document(doc))
        previous = self.meta.find_by_path(collection_id, path)
        if previous is not None:
            doc = self.meta.update_document(previous.id, status=DocumentStatus.ready, error=None, chunk_count=2,
                                            indexed_at=now(), **fields)
            return FakeResult(doc, replaced_doc_id=previous.id, replaced_title=previous.title)
        doc = Document(id=uuid.uuid4().hex, collection_id=collection_id, status=DocumentStatus.ready, chunk_count=2,
                       indexed_at=now(), **fields)
        return FakeResult(self.meta.create_document(doc))

    def reindex(self, doc_id: str, *, job_id: str | None = None,
                progress: Callable[[dict[str, Any]], None] | None = None) -> FakeResult:
        doc = self.meta.get_document(doc_id)
        if doc is None:
            raise LookupError("Document not found.")
        self.calls.append({"reindex": doc_id, "job_id": job_id})
        self._wait(doc.filename, job_id, progress)
        return FakeResult(self.meta.update_document(doc_id, status=DocumentStatus.ready, error=None,
                                                    chunk_count=2, indexed_at=now()))

    def delete(self, doc_id: str) -> None:
        if self.delete_gate is not None:
            self.delete_gate.wait(10)
        self.meta.delete_document(doc_id)

    def delete_collection(self, collection_id: str) -> None:
        self.meta.delete_collection(collection_id)


def add_doc(meta: MemoryMeta, collection_id: str, name: str, status: DocumentStatus = DocumentStatus.ready,
            **fields: Any) -> Document:
    data = name.encode()
    values: dict[str, Any] = {"title": name.rsplit(".", 1)[0], "filename": name, "source_path": name,
                              "media_type": "text/plain", "size_bytes": len(data), "content_hash": sha(data),
                              "uploaded_at": now(), "chunk_count": 2 if status == DocumentStatus.ready else 0}
    values.update(fields)
    return meta.create_document(Document(id=uuid.uuid4().hex, collection_id=collection_id, status=status,
                                         **values))


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("RAGKB_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("RAGKB_PROVIDER", "extractive")
    # The reranker is on by default; these tests never download its model.
    monkeypatch.setenv("RAGKB_RERANKER", "none")
    return Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        PROVIDER="extractive",
        RERANKER="none",
        MCP_ENABLED=False,
        MAX_UPLOAD_MB=1,
        LOG_LEVEL="WARNING",
    )


@contextmanager
def fake_server(settings: Settings, *, workers: int = 2, max_mb: int = 5, public: bool = False,
                meta: MemoryMeta | None = None, base_url: str = LOCAL_BASE) -> Iterator[SimpleNamespace]:
    """The real app and container over MemoryMeta and FakePipeline."""
    meta = meta or MemoryMeta()
    pipeline = FakePipeline(meta)
    configured = settings.model_copy(update={"INGEST_WORKERS": workers, "MAX_UPLOAD_MB": max_mb,
                                             "API_PUBLIC": public})
    deps = Deps(settings=configured, meta=meta, pipeline=pipeline)
    app = create_app(configured, deps=deps)
    try:
        with TestClient(app, base_url=base_url, raise_server_exceptions=False) as client:
            yield SimpleNamespace(client=client, deps=deps, meta=meta, pipeline=pipeline)
    finally:
        pipeline.release_all()


@pytest.fixture
def kb(settings) -> Iterator[SimpleNamespace]:
    with fake_server(settings) as server:
        yield server


@pytest.fixture
def bare_client(settings):
    """An app with whatever parts are installed; works with none of them."""
    app = create_app(settings)
    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client(settings):
    _require(*CORE_PARTS)
    if not PART_A:
        pytest.skip(PART_A_REASON)
    app = create_app(settings)
    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def ask_client(client):
    _require(*ANSWER_PARTS)
    return client


# ---------------------------------------------------------------- helpers


def assert_envelope(response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    message = body["error"]["message"]
    assert isinstance(message, str) and message
    assert "Traceback" not in message and ":\\" not in message
    return body["error"]


def make_collection(client: TestClient, name: str = "Handbook") -> str:
    response = client.post("/api/collections", json={"name": name, "description": "Company rules"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def upload(client: TestClient, cid: str, files: list[tuple[str, bytes]], title: str | None = None,
           relative_paths: list[str] | None = None):
    parts = [("files", (name, data, "application/octet-stream")) for name, data in files]
    data: dict[str, Any] = {}
    if title:
        data["title"] = title
    if relative_paths is not None:
        data["relative_paths"] = relative_paths
    return client.post(f"/api/collections/{cid}/documents", files=parts, data=data or None)


def wait_for_job(client: TestClient, job_id: str, timeout_s: float = 60.0,
                 statuses: tuple[str, ...] = FINISHED) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job did not reach {statuses}: {job}")
        time.sleep(0.02)


def wait_until(check: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.02)


def ingest(client: TestClient, cid: str, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    response = upload(client, cid, files)
    assert response.status_code == 202, response.text
    return [wait_for_job(client, job["id"]) for job in response.json()["jobs"]]


def first_job(response) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    return response.json()["jobs"][0]


def zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


# ---------------------------------------------------------------- envelope and basics (no parts needed)


def test_health_answers_even_without_parts(bare_client):
    body = bare_client.get("/api/health").json()
    assert body["status"] in ("ok", "degraded")
    for key in ("version", "knowledge_version", "counts", "store", "data_dir", "mcp", "max_upload_mb"):
        assert key in body
    assert body["view"] == "full"


def test_error_envelope_for_unknown_route_and_bad_body(bare_client):
    assert_envelope(bare_client.get("/api/does-not-exist"), 404, "not_found")
    assert_envelope(bare_client.post("/api/collections", content=b"{not json",
                                     headers={"content-type": "application/json"}), 422, "invalid_request")


def test_unexpected_error_is_500_internal_without_a_trace(settings):
    app = create_app(settings)

    @app.get("/api/boom")
    def boom() -> None:
        raise RuntimeError("secret detail at C:\\private\\file.txt")

    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        error = assert_envelope(client.get("/api/boom"), 500, "internal")
    assert "secret" not in error["message"] and "RuntimeError" not in error["message"]


def test_cross_site_writes_are_refused(bare_client):
    response = bare_client.post("/api/collections", json={"name": "x"}, headers={"origin": "https://evil.example"})
    assert_envelope(response, 403, "forbidden")


def test_api_answers_only_loopback_hosts_unless_public(settings):
    with TestClient(create_app(settings), base_url=REMOTE_BASE, raise_server_exceptions=False) as remote:
        error = assert_envelope(remote.get("/api/collections"), 403, "forbidden")
        assert "RAGKB_API_PUBLIC" in error["message"]
        assert_envelope(remote.post("/api/collections", json={"name": "x"}), 403, "forbidden")
        assert_envelope(remote.get("/api"), 403, "forbidden")
        assert_envelope(remote.get("/api/does-not-exist"), 403, "forbidden")
        health = remote.get("/api/health")
        assert health.status_code == 200 and health.json()["api"] == {"public": False}
    for local_base in ("http://localhost:8765", "http://[::1]:8765"):
        with TestClient(create_app(settings), base_url=local_base, raise_server_exceptions=False) as local:
            assert_envelope(local.get("/api/does-not-exist"), 404, "not_found")
    public = create_app(settings.model_copy(update={"API_PUBLIC": True}))
    with TestClient(public, base_url=REMOTE_BASE, raise_server_exceptions=False) as remote:
        assert_envelope(remote.get("/api/does-not-exist"), 404, "not_found")
        assert remote.get("/api/health").json()["api"] == {"public": True}


def test_health_hides_local_details_from_other_hosts_and_docs_need_loopback(settings):
    data_dir = str(settings.DATA_DIR.resolve())
    with TestClient(create_app(settings), base_url=REMOTE_BASE, raise_server_exceptions=False) as remote:
        response = remote.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["view"] == "public"
        assert "data_dir" not in body and "supported_types" not in body and "embed_model" not in body
        assert data_dir not in response.text
        assert {"status", "version", "counts", "store"} <= set(body)
        for path in ("/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"):
            assert_envelope(remote.get(path), 403, "forbidden")
    with TestClient(create_app(settings), base_url=LOCAL_BASE, raise_server_exceptions=False) as local:
        assert local.get("/api/health").json()["data_dir"] == data_dir
        assert local.get("/openapi.json").status_code == 200
        assert local.get("/docs").status_code == 200
    public = create_app(settings.model_copy(update={"API_PUBLIC": True}))
    with TestClient(public, base_url=REMOTE_BASE, raise_server_exceptions=False) as remote:
        assert remote.get("/openapi.json").status_code == 200
        assert "data_dir" not in remote.get("/api/health").json()


def test_an_unknown_embedding_model_is_reported_plainly(settings, monkeypatch):
    _require("ragkb.retrieval.embedder")
    monkeypatch.delenv("RAGKB_FAKE_EMBEDDER")
    deps = build_deps(settings.model_copy(update={"EMBED_MODEL": "intfloat/multilingual-e5-small"}))
    try:
        assert deps.embedder is None and deps.retriever is None
        reason = deps.problems["embedder"]
        assert "is not supported" in reason and "paraphrase-multilingual-MiniLM-L12-v2" in reason
        health = deps.health()
        assert health["status"] == "degraded" and any("is not supported" in p for p in health["problems"])
    finally:
        deps.close()


def test_rate_limiter_counts_per_key():
    limiter = RateLimiter(2, 60)
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    wait = limiter.check("a")
    assert wait is not None and 0 < wait <= 60
    assert limiter.check("b") is None


# ---------------------------------------------------------------- errors and names


def test_malformed_bodies_get_a_plain_sentence_with_the_fields(kb):
    expected = "The request body must be a JSON object with these fields: name (required), description."
    for content in (b'"hello"', b"[1, 2]", b"null", b""):
        response = kb.client.post("/api/collections", content=content, headers={"content-type": "application/json"})
        error = assert_envelope(response, 422, "invalid_request")
        assert error["message"] == expected, content
    broken = kb.client.post("/api/collections", content=b"{bad", headers={"content-type": "application/json"})
    message = assert_envelope(broken, 422, "invalid_request")["message"]
    assert message.startswith("The request body is not valid JSON.") and "name (required)" in message
    cid = make_collection(kb.client)
    ask = kb.client.post(f"/api/collections/{cid}/ask", content=b"[]", headers={"content-type": "application/json"})
    message = assert_envelope(ask, 422, "invalid_request")["message"]
    assert message == ("The request body must be a JSON object with these fields: question (required), top_k, "
                       "provider, model.")
    missing = assert_envelope(kb.client.post("/api/collections", json={}), 422, "invalid_request")
    assert missing["message"] == "name is required."
    blank = assert_envelope(kb.client.post("/api/collections", json={"name": "  "}), 422, "invalid_request")
    assert "valid dictionary" not in blank["message"] and not blank["message"].startswith("Value error")


def test_unsupported_type_message_starts_plainly_and_shortens_the_name(kb):
    cid = make_collection(kb.client)
    name = "x" * 300 + ".exe"
    error = assert_envelope(upload(kb.client, cid, [(name, b"MZ binary")]), 415, "unsupported_file_type")
    message = error["message"]
    assert message.startswith("Files of type .exe are not supported (")
    assert "x" * 61 not in message and "...x" in message and ".exe)" in message
    assert "Supported types:" in message and ".zip" in message


def test_collection_names_and_descriptions_are_cleaned(kb):
    created = kb.client.post("/api/collections", json={
        "name": "abc" + RLO + "def" + chr(10) + chr(0) + " team" + chr(0x2066),
        "description": "line one" + chr(10) + "line" + chr(9) + "two" + chr(0x85) + chr(0x7F),
    })
    assert created.status_code == 201, created.text
    assert created.json()["name"] == "abcdef team"
    assert created.json()["description"] == "line one line two"
    for name in (chr(0) + chr(10), RLO + chr(0x200E) + " ", chr(0x9B)):
        assert_envelope(kb.client.post("/api/collections", json={"name": name}), 422, "invalid_request")
    duplicate = kb.client.post("/api/collections", json={"name": "ABC" + chr(0x200F) + "DEF   TEAM"})
    assert_envelope(duplicate, 409, "conflict")


def test_upload_file_names_and_titles_are_cleaned(kb):
    cid = make_collection(kb.client)
    # The client percent-encodes ASCII controls in file names, so a C1 control stands in for them here.
    job = first_job(upload(kb.client, cid, [("re" + RLO + "port" + chr(0x9B) + ".md", MARKDOWN)],
                           title="Leave" + chr(0x1B) + " rules" + chr(9) + "2026" + RLO))
    done = wait_for_job(kb.client, job["id"], 10)
    assert done["filename"] == "report.md" and done["status"] == "ready"
    assert kb.pipeline.calls[-1]["filename"] == "report.md"
    assert kb.pipeline.calls[-1]["title"] == "Leave rules 2026"
    error = assert_envelope(upload(kb.client, cid, [("notes.md", TEXT)], title=chr(0x202A) + chr(0x7F)), 400,
                            "invalid_request")
    assert "title" in error["message"]


# ---------------------------------------------------------------- collections and counts


def test_collections_crud(kb):
    client = kb.client
    cid = make_collection(client)
    listed = client.get("/api/collections").json()
    assert [c["id"] for c in listed] == [cid]
    assert listed[0]["name"] == "Handbook" and listed[0]["documents"] == 0 and listed[0]["documents_total"] == 0

    assert_envelope(client.post("/api/collections", json={"name": "handbook"}), 409, "conflict")
    assert_envelope(client.post("/api/collections", json={"name": "   "}), 422, "invalid_request")

    stats = client.get(f"/api/collections/{cid}/stats").json()
    assert stats["documents"] == 0 and stats["documents_ready"] == 0 and stats["documents_by_status"] == {}

    deleted = client.delete(f"/api/collections/{cid}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] == cid
    assert client.get("/api/collections").json() == []
    assert_envelope(client.get(f"/api/collections/{cid}/documents"), 404, "not_found")
    assert_envelope(client.delete(f"/api/collections/{cid}"), 404, "not_found")


def test_document_counts_put_ready_first(kb):
    cid = make_collection(kb.client)
    other = make_collection(kb.client, "Other")
    for name, status in (("a.md", DocumentStatus.ready), ("b.md", DocumentStatus.ready),
                         ("c.md", DocumentStatus.failed), ("d.md", DocumentStatus.processing),
                         ("e.md", DocumentStatus.queued)):
        add_doc(kb.meta, cid, name, status)
    add_doc(kb.meta, other, "z.md", DocumentStatus.cancelled)
    by_status = {"failed": 1, "processing": 1, "queued": 1, "ready": 2}

    listed = {c["id"]: c for c in kb.client.get("/api/collections").json()}
    assert listed[cid]["documents"] == 2 and listed[cid]["documents_total"] == 5
    assert listed[cid]["documents_by_status"] == by_status
    assert listed[other]["documents"] == 0 and listed[other]["documents_total"] == 1

    stats = kb.client.get(f"/api/collections/{cid}/stats").json()
    assert stats["documents"] == stats["documents_ready"] == 2
    assert stats["documents_total"] == 5 and stats["documents_by_status"] == by_status

    counts = kb.client.get("/api/health").json()["counts"]
    assert counts["documents"] == counts["documents_ready"] == 2
    assert counts["documents_failed"] == 1 and counts["documents_processing"] == 2
    assert counts["documents_total"] == 6 and counts["collections"] == 2


# ---------------------------------------------------------------- uploads: per-file outcomes, folders, zips


def test_each_file_is_either_a_job_or_a_rejection(kb):
    cid = make_collection(kb.client)
    too_big = b"word " * (5 * MB // 5 + 10)
    response = upload(kb.client, cid, [("leave.md", MARKDOWN), ("run.sh", b"echo"), ("empty.txt", b""),
                                       ("big.txt", too_big)])
    assert response.status_code == 202, response.text
    body = response.json()
    assert [j["filename"] for j in body["jobs"]] == ["leave.md"]
    assert [(r["filename"], r["code"]) for r in body["rejected"]] == [
        ("run.sh", "unsupported_file_type"), ("empty.txt", "empty_document"), ("big.txt", "file_too_large")]
    assert all(r["message"] for r in body["rejected"])
    wait_for_job(kb.client, body["jobs"][0]["id"], 10)
    assert [d["filename"] for d in kb.client.get(f"/api/collections/{cid}/documents").json()] == ["leave.md"]

    only_bad = assert_envelope(upload(kb.client, cid, [("a.exe", b"MZ"), ("b.exe", b"MZ")]), 415,
                               "unsupported_file_type")
    assert only_bad["message"].startswith("None of the 2 files could be added.")
    assert_envelope(upload(kb.client, cid, [("tool.exe", b"MZ binary")]), 415, "unsupported_file_type")
    assert_envelope(upload(kb.client, cid, [("big.txt", too_big)]), 413, "file_too_large")
    assert_envelope(upload(kb.client, cid, [("empty.txt", b"")]), 400, "empty_document")
    assert_envelope(upload(kb.client, cid, [("a.md", MARKDOWN), ("b.txt", TEXT)], title="One title"), 400,
                    "invalid_request")
    assert_envelope(upload(kb.client, "no-such-collection", [("a.md", MARKDOWN)]), 404, "not_found")


def test_a_password_protected_file_is_rejected_up_front_and_leaves_no_row(kb):
    cid = make_collection(kb.client)
    locked_pdf = encrypted_pdf(build_pdf([["Secret text for the test."]]))
    locked_docx = bytes.fromhex("d0cf11e0a1b11ae1") + bytes(504) + "EncryptedPackage".encode("utf-16-le")
    error = assert_envelope(upload(kb.client, cid, [("locked.pdf", locked_pdf)]), 400, "encrypted_file")
    assert error["message"] == "The PDF is password protected. Remove the password and upload it again (locked.pdf)."

    response = upload(kb.client, cid, [("leave.md", MARKDOWN), ("locked.docx", locked_docx)])
    assert response.status_code == 202, response.text
    body = response.json()
    assert [j["filename"] for j in body["jobs"]] == ["leave.md"]
    assert [(r["filename"], r["code"]) for r in body["rejected"]] == [("locked.docx", "encrypted_file")]
    assert "Word file is password protected" in body["rejected"][0]["message"]

    archive = zip_bytes([("handbook/locked.pdf", locked_pdf), ("handbook/expenses.txt", TEXT)])
    body = upload(kb.client, cid, [("corpus.zip", archive)]).json()
    assert [j["source_path"] for j in body["jobs"]] == ["handbook/expenses.txt"]
    assert [(r["filename"], r["source_path"], r["code"]) for r in body["rejected"]] == [
        ("locked.pdf", "handbook/locked.pdf", "encrypted_file")]

    for job in kb.client.get(f"/api/collections/{cid}/jobs").json():
        wait_for_job(kb.client, job["id"], 10)
    jobs = kb.client.get(f"/api/collections/{cid}/jobs").json()
    assert sorted(j["filename"] for j in jobs) == ["expenses.txt", "leave.md"], "no job for a locked file"
    docs = kb.client.get(f"/api/collections/{cid}/documents").json()
    assert sorted(d["filename"] for d in docs) == ["expenses.txt", "leave.md"], "no document for a locked file"
    assert sorted(c["filename"] for c in kb.pipeline.calls) == ["expenses.txt", "leave.md"]


def test_failed_indexing_is_a_failed_job_not_a_rejection(kb):
    cid = make_collection(kb.client)
    job = first_job(upload(kb.client, cid, [("broken-scan.md", b"# unreadable")]))
    done = wait_for_job(kb.client, job["id"], 10)
    assert done["status"] == "failed" and done["error"] == "The file could not be read."
    assert done["doc_id"] and done["stage"] == "failed"


def test_folder_upload_passes_relative_paths(kb):
    cid = make_collection(kb.client)
    response = upload(kb.client, cid, [("a.md", MARKDOWN), ("b.txt", TEXT), ("c.md", b"# C\n\nText.\n")],
                      relative_paths=["team/a.md", "team" + chr(0) + "/sub/./b.txt", "../c.md"])
    assert response.status_code == 202, response.text
    body = response.json()
    assert [j["source_path"] for j in body["jobs"]] == ["team/a.md", "team/sub/b.txt"]
    assert [(r["filename"], r["code"]) for r in body["rejected"]] == [("c.md", "invalid_request")]
    for job in body["jobs"]:
        assert wait_for_job(kb.client, job["id"], 10)["status"] == "ready"
    assert sorted(c["source_path"] for c in kb.pipeline.calls) == ["team/a.md", "team/sub/b.txt"]
    docs = {d["source_path"] for d in kb.client.get(f"/api/collections/{cid}/documents").json()}
    assert docs == {"team/a.md", "team/sub/b.txt"}
    mismatch = upload(kb.client, cid, [("d.md", MARKDOWN)], relative_paths=["x/d.md", "x/e.md"])
    assert "relative_paths" in assert_envelope(mismatch, 400, "invalid_request")["message"]


def test_zip_upload_expands_entries_and_refuses_traversal(kb):
    cid = make_collection(kb.client)
    archive = zip_bytes([
        ("handbook/", b""),
        ("handbook/leave.md", MARKDOWN),
        ("handbook/hr/expenses.txt", TEXT),
        ("../escape.md", b"# Escape\n\nOutside the folder.\n"),
        ("/absolute.md", b"# Absolute\n"),
        ("handbook/../../up.md", b"# Up\n"),
        ("handbook/.DS_Store", b"junk"),
        ("__MACOSX/handbook/._leave.md", b"junk"),
        ("handbook/tool.exe", b"MZ"),
        ("handbook/empty.md", b""),
        ("handbook/inner.zip", b"PK"),
    ])
    response = upload(kb.client, cid, [("corpus.zip", archive)])
    assert response.status_code == 202, response.text
    body = response.json()
    assert [(j["filename"], j["source_path"]) for j in body["jobs"]] == [
        ("leave.md", "handbook/leave.md"), ("expenses.txt", "handbook/hr/expenses.txt")]
    refused = {(r["filename"], r["code"]) for r in body["rejected"]}
    assert refused == {
        ("escape.md", "invalid_request"), ("absolute.md", "invalid_request"), ("up.md", "invalid_request"),
        ("tool.exe", "unsupported_file_type"), ("empty.md", "empty_document"),
        ("inner.zip", "unsupported_file_type"),
    }
    assert body["skipped"] == 3
    for job in body["jobs"]:
        assert wait_for_job(kb.client, job["id"], 10)["status"] == "ready"
    assert sorted(c["source_path"] for c in kb.pipeline.calls) == ["handbook/hr/expenses.txt", "handbook/leave.md"]
    assert all("escape" not in c["filename"] for c in kb.pipeline.calls)
    assert_envelope(upload(kb.client, cid, [("corpus.zip", archive)], title="A title"), 400, "invalid_request")


def test_zip_archive_limits_refuse_the_whole_archive(settings):
    with fake_server(settings, max_mb=1) as kb:
        cid = make_collection(kb.client)
        entry = b"a" * 1_000_000
        bomb = zip_bytes([(f"part-{i}.txt", entry) for i in range(11)])
        error = assert_envelope(upload(kb.client, cid, [("bomb.zip", bomb)]), 413, "file_too_large")
        assert "expands to more than the 10 MB limit" in error["message"]
        assert kb.pipeline.calls == []
        damaged = assert_envelope(upload(kb.client, cid, [("fake.zip", b"not a zip at all")]), 400,
                                  "invalid_request")
        assert "could not be opened" in damaged["message"]
        nothing = assert_envelope(upload(kb.client, cid, [("folders.zip", zip_bytes([("only/", b"")]))]), 400,
                                  "invalid_request")
        assert "no files" in nothing["message"]


# ---------------------------------------------------------------- jobs: lanes, labels, progress, cancel


def test_small_files_do_not_wait_behind_large_ones(kb):
    cid = make_collection(kb.client)
    big = b"a" * (2 * MB)
    kb.pipeline.hold("big-one.txt")
    kb.pipeline.hold("big-two.txt")
    first = first_job(upload(kb.client, cid, [("big-one.txt", big)]))
    assert kb.pipeline.started["big-one.txt"].wait(5)
    second = first_job(upload(kb.client, cid, [("big-two.txt", big + b"b")]))
    small = first_job(upload(kb.client, cid, [("small.txt", b"A small file that must not wait.")]))

    assert wait_for_job(kb.client, small["id"], 5)["status"] == "ready"
    assert kb.client.get(f"/api/jobs/{first['id']}").json()["status"] == "processing"
    waiting = kb.client.get(f"/api/jobs/{second['id']}").json()
    assert waiting["status"] == "queued" and waiting["stage"] == "queued"

    kb.pipeline.release_all()
    assert wait_for_job(kb.client, first["id"], 10)["status"] == "ready"
    assert wait_for_job(kb.client, second["id"], 10)["status"] == "ready"


def test_uploads_and_reindexes_read_queued_until_a_worker_starts_them(settings):
    with fake_server(settings, workers=1) as kb:
        cid = make_collection(kb.client)
        manual = add_doc(kb.meta, cid, "manual.txt")
        other = add_doc(kb.meta, cid, "other.txt")
        kb.pipeline.hold("manual.txt")
        running = kb.client.post(f"/api/documents/{manual.id}/reindex")
        assert running.status_code == 202 and running.json()["kind"] == "reindex"
        assert kb.pipeline.started["manual.txt"].wait(5)
        assert kb.client.get(f"/api/jobs/{running.json()['id']}").json()["status"] == "processing"

        waiting_upload = first_job(upload(kb.client, cid, [("next.txt", TEXT)]))
        waiting_reindex = kb.client.post(f"/api/documents/{other.id}/reindex").json()
        assert waiting_upload["status"] == "queued" and waiting_reindex["status"] == "queued"
        for job_id in (waiting_upload["id"], waiting_reindex["id"]):
            job = kb.client.get(f"/api/jobs/{job_id}").json()
            assert (job["status"], job["stage"]) == ("queued", "queued")
        busy = kb.client.post(f"/api/documents/{manual.id}/reindex")
        assert_envelope(busy, 409, "conflict")

        kb.pipeline.release_all()
        for job_id in (running.json()["id"], waiting_upload["id"], waiting_reindex["id"]):
            assert wait_for_job(kb.client, job_id, 10)["status"] == "ready"


def test_progress_is_written_to_the_job(kb):
    cid = make_collection(kb.client)
    kb.pipeline.steps["slow.txt"] = [
        {"stage": "parsing", "chunks_done": 0, "chunks_total": None, "eta_seconds": None},
        {"stage": "embedding", "chunks_done": 3, "chunks_total": 10, "eta_seconds": 4},
    ]
    kb.pipeline.hold("slow.txt")
    job = first_job(upload(kb.client, cid, [("slow.txt", TEXT)]))
    assert kb.pipeline.started["slow.txt"].wait(5)
    live = kb.client.get(f"/api/jobs/{job['id']}").json()
    assert live["status"] == "processing"
    assert (live["stage"], live["chunks_done"], live["chunks_total"], live["eta_seconds"]) == ("embedding", 3, 10, 4)
    assert live["bytes_total"] == len(TEXT)
    kb.pipeline.release_all()
    done = wait_for_job(kb.client, job["id"], 10)
    assert (done["status"], done["stage"], done["chunks_done"], done["chunks_total"]) == ("ready", "done", 2, 2)


def test_cancel_ends_jobs_as_cancelled(settings):
    with fake_server(settings, workers=1) as kb:
        cid = make_collection(kb.client)
        kb.pipeline.hold("long.txt")
        running = first_job(upload(kb.client, cid, [("long.txt", TEXT)]))
        assert kb.pipeline.started["long.txt"].wait(5)
        waiting = first_job(upload(kb.client, cid, [("waiting.txt", MARKDOWN)]))

        queued_cancel = kb.client.post(f"/api/jobs/{waiting['id']}/cancel")
        assert queued_cancel.status_code == 200
        assert queued_cancel.json()["status"] == "cancelled" and queued_cancel.json()["error"] is None
        assert not list((kb.deps.settings.uploads_dir / ".incoming").glob(waiting["id"]))

        response = kb.client.post(f"/api/jobs/{running['id']}/cancel")
        assert response.status_code == 200 and response.json()["cancel_requested"] is True
        ended = wait_for_job(kb.client, running["id"], 10)
        assert (ended["status"], ended["error"], ended["stage"]) == ("cancelled", None, "cancelled")
        assert kb.client.get(f"/api/collections/{cid}/documents").json() == []
        assert all(call["filename"] != "waiting.txt" for call in kb.pipeline.calls)

        finished = ingest(kb.client, cid, [("done.txt", b"Finished file.")])[0]
        again = kb.client.post(f"/api/jobs/{finished['id']}/cancel").json()
        assert again["status"] == "ready" and again["cancel_requested"] is False
        assert_envelope(kb.client.post("/api/jobs/nope/cancel"), 404, "not_found")


def test_deletes_do_not_wait_for_ingestion_and_are_bounded(kb):
    cid = make_collection(kb.client)
    keep = ingest(kb.client, cid, [("keep.txt", b"Kept for a moment.")])[0]
    slow = ingest(kb.client, cid, [("slow-delete.txt", b"Deleted slowly.")])[0]
    kb.pipeline.hold("huge.txt")
    kb.pipeline.hold("busy.txt")
    first_job(upload(kb.client, cid, [("huge.txt", b"h" * (2 * MB))]))
    first_job(upload(kb.client, cid, [("busy.txt", b"Small but held.")]))
    assert kb.pipeline.started["huge.txt"].wait(5) and kb.pipeline.started["busy.txt"].wait(5)

    started = time.monotonic()
    deleted = kb.client.delete(f"/api/documents/{keep['doc_id']}")
    assert deleted.status_code == 200 and time.monotonic() - started < 3
    assert_envelope(kb.client.get(f"/api/documents/{keep['doc_id']}"), 404, "not_found")

    kb.deps.delete_wait_s = 0.2
    kb.pipeline.delete_gate = threading.Event()
    started = time.monotonic()
    pending = kb.client.delete(f"/api/documents/{slow['doc_id']}")
    assert pending.status_code == 202 and pending.json()["pending"] is True
    assert time.monotonic() - started < 3
    kb.pipeline.delete_gate.set()
    wait_until(lambda: kb.client.get(f"/api/documents/{slow['doc_id']}").status_code == 404)


def test_deleting_a_document_cancels_its_waiting_jobs(settings):
    with fake_server(settings, workers=1) as kb:
        cid = make_collection(kb.client)
        doc = add_doc(kb.meta, cid, "target.txt")
        kb.pipeline.hold("blocker.txt")
        first_job(upload(kb.client, cid, [("blocker.txt", TEXT)]))
        assert kb.pipeline.started["blocker.txt"].wait(5)
        reindex = kb.client.post(f"/api/documents/{doc.id}/reindex").json()
        assert kb.client.delete(f"/api/documents/{doc.id}").status_code == 200
        assert kb.client.get(f"/api/jobs/{reindex['id']}").json()["status"] == "cancelled"
        kb.pipeline.release_all()


# ---------------------------------------------------------------- jobs: replaced, duplicates, retry, summary


def test_replaced_and_already_indexed_are_reported(kb):
    cid = make_collection(kb.client)
    first = ingest(kb.client, cid, [("policy.md", b"# Policy\n\nVersion one.\n")])[0]
    assert first["replaced_doc_id"] is None and first["already_indexed"] is False

    second = ingest(kb.client, cid, [("policy.md", b"# Policy\n\nVersion two.\n")])[0]
    assert second["status"] == "ready"
    assert second["replaced_doc_id"] == first["doc_id"] and second["replaced_title"] == "policy"

    again = first_job(upload(kb.client, cid, [("copy-of-policy.md", b"# Policy\n\nVersion two.\n")]))
    assert again["status"] == "ready" and again["already_indexed"] is True and again["doc_id"] == first["doc_id"]

    kb.pipeline.hold("dup.txt")
    same = b"The same bytes, uploaded twice at once."
    pair = upload(kb.client, cid, [("dup.txt", same), ("dup-copy.txt", same)]).json()["jobs"]
    copy = wait_for_job(kb.client, pair[1]["id"], 10)
    kb.pipeline.release_all()
    raced = wait_for_job(kb.client, pair[0]["id"], 10)
    assert raced["status"] == "ready" and raced["already_indexed"] is True and raced["doc_id"] == copy["doc_id"]


def test_retry_reindexes_a_failed_document_and_clears_its_error(settings):
    with fake_server(settings, workers=1) as kb:
        cid = make_collection(kb.client)
        failed = add_doc(kb.meta, cid, "scan.txt", DocumentStatus.failed, error="The file could not be read.")
        ready = add_doc(kb.meta, cid, "fine.txt")
        kb.pipeline.hold("blocker.txt")
        first_job(upload(kb.client, cid, [("blocker.txt", TEXT)]))
        assert kb.pipeline.started["blocker.txt"].wait(5)

        response = kb.client.post(f"/api/documents/{failed.id}/retry")
        assert response.status_code == 202
        job = response.json()
        assert job["kind"] == "retry" and job["status"] == "queued" and job["doc_id"] == failed.id
        waiting = kb.client.get(f"/api/documents/{failed.id}").json()
        assert waiting["status"] == "queued" and waiting["error"] is None

        cancelled = kb.client.post(f"/api/jobs/{job['id']}/cancel").json()
        assert cancelled["status"] == "cancelled"
        restored = kb.client.get(f"/api/documents/{failed.id}").json()
        assert restored["status"] == "failed" and restored["error"] == "The file could not be read."

        retried = kb.client.post(f"/api/documents/{failed.id}/retry").json()
        kb.pipeline.release_all()
        assert wait_for_job(kb.client, retried["id"], 10)["status"] == "ready"
        healed = kb.client.get(f"/api/documents/{failed.id}").json()
        assert healed["status"] == "ready" and healed["error"] is None

        assert_envelope(kb.client.post(f"/api/documents/{ready.id}/retry"), 409, "conflict")
        assert_envelope(kb.client.post("/api/documents/nope/retry"), 404, "not_found")


def test_jobs_summary_counts_each_job_once(kb):
    cid = make_collection(kb.client)
    other = make_collection(kb.client, "Other")
    rows: list[tuple[DocumentStatus, dict[str, Any]]] = [
        (DocumentStatus.queued, {}), (DocumentStatus.processing, {}), (DocumentStatus.ready, {}),
        (DocumentStatus.ready, {"already_indexed": True}), (DocumentStatus.ready, {"replaced_doc_id": "old"}),
        (DocumentStatus.failed, {}), (DocumentStatus.cancelled, {}),
    ]
    for status, fields in rows:
        job = kb.meta.create_job(cid, "")
        kb.meta.update_job(job.id, status, **fields)
    old = kb.meta.create_job(cid, "")
    kb.meta.update_job(old.id, DocumentStatus.ready, created_at=now() - timedelta(days=2))
    kb.meta.update_job(kb.meta.create_job(other, "").id, DocumentStatus.failed)

    summary = kb.client.get(f"/api/collections/{cid}/jobs/summary").json()
    expected = {"queued": 1, "processing": 1, "ready": 1, "failed": 1, "cancelled": 1, "replaced": 1,
                "duplicates": 1, "total": 7}
    assert {k: summary[k] for k in expected} == expected
    wider = kb.client.get(f"/api/collections/{cid}/jobs/summary",
                          params={"since": (now() - timedelta(days=3)).isoformat()}).json()
    assert wider["ready"] == 2 and wider["total"] == 8
    later = kb.client.get(f"/api/collections/{cid}/jobs/summary",
                          params={"since": (now() + timedelta(hours=1)).isoformat()}).json()
    assert later["total"] == 0
    assert_envelope(kb.client.get("/api/collections/nope/jobs/summary"), 404, "not_found")


def test_jobs_left_by_a_stopped_server_are_closed_at_startup(settings):
    meta = MemoryMeta()
    collection = meta.create_collection("Handbook")
    stuck = add_doc(meta, collection.id, "stuck.txt", DocumentStatus.processing)
    running = meta.create_job(collection.id, stuck.id)
    meta.update_job(running.id, DocumentStatus.processing, stage="embedding")
    waiting = meta.create_job(collection.id, "")
    finished = meta.create_job(collection.id, "")
    meta.update_job(finished.id, DocumentStatus.ready)
    spool = settings.uploads_dir / ".incoming"
    spool.mkdir(parents=True, exist_ok=True)
    (spool / waiting.id).write_bytes(b"left behind")

    with fake_server(settings, meta=meta) as kb:
        for job_id in (running.id, waiting.id):
            job = kb.client.get(f"/api/jobs/{job_id}").json()
            assert (job["status"], job["error"]) == ("failed", INTERRUPTED_MESSAGE)
        assert kb.client.get(f"/api/jobs/{finished.id}").json()["status"] == "ready"
        doc = kb.client.get(f"/api/documents/{stuck.id}").json()
        assert doc["status"] == "failed" and doc["error"] == INTERRUPTED_MESSAGE
        assert not (spool / waiting.id).exists()


# ---------------------------------------------------------------- feedback


def test_feedback_can_be_read_back(kb):
    cid = make_collection(kb.client)
    other = make_collection(kb.client, "Other")
    answers = []
    for collection_id, question in ((cid, "How many leave days?"), (cid, "Who approves expenses?"),
                                    (other, "Where is parking?")):
        answer = Answer(id=uuid.uuid4().hex, question=question, collection_id=collection_id,
                        status=AnswerStatus.answered, answer="Twenty five [1].", created_at=now())
        kb.meta.save_answer(answer.model_dump_json(), answer.id, collection_id)
        answers.append(answer)

    url = f"/api/answers/{answers[0].id}/feedback"
    saved = kb.client.post(url, json={"rating": "up", "note": "  Clear and correct  "})
    assert saved.status_code == 201
    assert saved.json()["id"] and saved.json()["question"] == "How many leave days?"
    assert saved.json()["collection_id"] == cid and saved.json()["note"] == "Clear and correct"
    assert kb.client.post(url, json={"rating": "down"}).status_code == 201
    assert kb.client.post(f"/api/answers/{answers[1].id}/feedback", json={"rating": "down"}).status_code == 201
    assert kb.client.post(f"/api/answers/{answers[2].id}/feedback", json={"rating": "up"}).status_code == 201

    listed = kb.client.get(f"/api/collections/{cid}/feedback").json()
    assert [(f["question"], f["rating"]) for f in listed] == [
        ("Who approves expenses?", "down"), ("How many leave days?", "down"), ("How many leave days?", "up")]
    assert {"id", "answer_id", "question", "rating", "note", "created_at"} <= set(listed[0])
    assert len(kb.client.get(f"/api/collections/{cid}/feedback", params={"limit": 1}).json()) == 1

    one = kb.client.get(url).json()
    assert [f["rating"] for f in one] == ["down", "up"] and {f["answer_id"] for f in one} == {answers[0].id}
    assert_envelope(kb.client.post(url, json={"rating": "meh"}), 422, "invalid_request")
    assert_envelope(kb.client.get("/api/collections/nope/feedback"), 404, "not_found")
    assert_envelope(kb.client.get("/api/answers/nope/feedback"), 404, "not_found")
    assert_envelope(kb.client.post("/api/answers/nope/feedback", json={"rating": "up"}), 404, "not_found")


def test_answers_retrieve_with_the_composer_budget_and_search_does_not(kb):
    _require(*ANSWER_PARTS)
    budgets: list[int | None] = []

    class Recorder:
        def search(self, collection_id: str, query: str, top_k: int, *, budget_chars: int | None = None) -> list[Any]:
            budgets.append(budget_chars)
            return []

    kb.deps.retriever = Recorder()
    cid = make_collection(kb.client)
    assert kb.client.post(f"/api/collections/{cid}/search", json={"query": "leave"}).json() == []
    body = kb.client.post(f"/api/collections/{cid}/ask", json={"question": "Annual leave?"}).json()
    assert body["answer"] is None
    assert budgets == [None, kb.deps.settings.MAX_CONTEXT_CHARS]


# ---------------------------------------------------------------- end to end on the real stores (part A)


def test_deleting_a_collection_removes_its_documents(client):
    cid = make_collection(client)
    jobs = ingest(client, cid, [("leave.md", MARKDOWN)])
    doc_id = jobs[0]["doc_id"]
    assert client.delete(f"/api/collections/{cid}").status_code == 200
    assert_envelope(client.get(f"/api/documents/{doc_id}"), 404, "not_found")


def test_upload_markdown_and_text_reach_ready(client):
    cid = make_collection(client)
    response = upload(client, cid, [("leave-policy.md", MARKDOWN), ("expenses.txt", TEXT)])
    assert response.status_code == 202, response.text
    jobs = response.json()["jobs"]
    assert len(jobs) == 2 and response.json()["rejected"] == []
    assert all(j["status"] == "queued" and j["already_indexed"] is False for j in jobs)
    assert [j["filename"] for j in jobs] == ["leave-policy.md", "expenses.txt"]

    finished = [wait_for_job(client, j["id"]) for j in jobs]
    assert [j["status"] for j in finished] == ["ready", "ready"], finished
    assert all(j["doc_id"] for j in finished)
    assert all(j["finished_at"] for j in finished)
    assert all(j["stage"] == "done" for j in finished)

    docs = client.get(f"/api/collections/{cid}/documents").json()
    assert len(docs) == 2
    by_name = {d["filename"]: d for d in docs}
    md = by_name["leave-policy.md"]
    assert md["status"] == "ready" and md["chunk_count"] >= 1 and md["media_type"] == "text/markdown"
    assert md["size_bytes"] == len(MARKDOWN) and md["indexed_at"]

    one = client.get(f"/api/documents/{md['id']}").json()
    assert one["id"] == md["id"]

    chunks = client.get(f"/api/documents/{md['id']}/chunks").json()
    assert len(chunks) == md["chunk_count"]
    assert [c["ordinal"] for c in chunks] == sorted(c["ordinal"] for c in chunks)
    assert all(c["doc_id"] == md["id"] and c["collection_id"] == cid for c in chunks)
    assert any("25 days" in c["text"] for c in chunks)

    original = client.get(f"/api/documents/{md['id']}/file")
    assert original.status_code == 200
    assert original.content == MARKDOWN
    assert "attachment" in original.headers["content-disposition"]

    jobs_listed = client.get(f"/api/collections/{cid}/jobs").json()
    assert {j["id"] for j in jobs} <= {j["id"] for j in jobs_listed}

    stats = client.get(f"/api/collections/{cid}/stats").json()
    assert stats["documents"] == 2 and stats["chunks"] >= 2
    assert stats["documents_by_status"] == {"ready": 2}
    assert client.get("/api/collections").json()[0]["documents"] == 2


def test_upload_job_keeps_its_document_after_a_restart(settings):
    _require(*CORE_PARTS)
    if not PART_A:
        pytest.skip(PART_A_REASON)
    with TestClient(create_app(settings), base_url=LOCAL_BASE, raise_server_exceptions=False) as first:
        cid = make_collection(first)
        job = ingest(first, cid, [("leave.md", MARKDOWN)])[0]
        assert job["status"] == "ready" and job["doc_id"]
    # A new app on the same data folder has no in-memory job details: the doc id must come from the store.
    with TestClient(create_app(settings), base_url=LOCAL_BASE, raise_server_exceptions=False) as second:
        again = second.get(f"/api/jobs/{job['id']}").json()
        assert again["doc_id"] == job["doc_id"]
        listed = second.get(f"/api/collections/{cid}/jobs").json()
        assert [j["doc_id"] for j in listed if j["id"] == job["id"]] == [job["doc_id"]]
        assert second.get(f"/api/documents/{job['doc_id']}").status_code == 200


def test_same_content_is_reported_as_already_indexed(client):
    cid = make_collection(client)
    first = ingest(client, cid, [("leave.md", MARKDOWN)])[0]
    again = upload(client, cid, [("copy-of-leave.md", MARKDOWN)])
    assert again.status_code == 202
    job = again.json()["jobs"][0]
    assert job["already_indexed"] is True
    assert job["status"] == "ready"
    assert job["doc_id"] == first["doc_id"]
    assert len(client.get(f"/api/collections/{cid}/documents").json()) == 1


def test_upload_with_title(client):
    cid = make_collection(client)
    job = ingest(client, cid, [("notes.txt", TEXT)])
    assert job[0]["status"] == "ready"
    titled = upload(client, cid, [("leave.md", MARKDOWN)], title="Leave rules 2026")
    assert titled.status_code == 202
    done = wait_for_job(client, titled.json()["jobs"][0]["id"])
    doc = client.get(f"/api/documents/{done['doc_id']}").json()
    assert doc["title"] == "Leave rules 2026"


def test_reindex_and_delete_document(client):
    cid = make_collection(client)
    doc_id = ingest(client, cid, [("leave.md", MARKDOWN)])[0]["doc_id"]

    response = client.post(f"/api/documents/{doc_id}/reindex")
    assert response.status_code == 202
    job = wait_for_job(client, response.json()["id"])
    assert job["status"] == "ready" and job["doc_id"] == doc_id and job["kind"] == "reindex"

    deleted = client.delete(f"/api/documents/{doc_id}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] == doc_id
    assert_envelope(client.get(f"/api/documents/{doc_id}"), 404, "not_found")
    assert_envelope(client.get(f"/api/documents/{doc_id}/chunks"), 404, "not_found")
    assert client.get(f"/api/collections/{cid}/documents").json() == []
    assert client.get(f"/api/collections/{cid}/stats").json()["chunks"] == 0
    assert_envelope(client.delete(f"/api/documents/{doc_id}"), 404, "not_found")
    assert_envelope(client.post(f"/api/documents/{doc_id}/reindex"), 404, "not_found")


def test_zip_upload_on_the_real_pipeline(client):
    cid = make_collection(client)
    archive = zip_bytes([("handbook/leave.md", MARKDOWN), ("handbook/hr/expenses.txt", TEXT)])
    jobs = [wait_for_job(client, j["id"]) for j in first_job_list(upload(client, cid, [("corpus.zip", archive)]))]
    assert [j["status"] for j in jobs] == ["ready", "ready"]
    docs = client.get(f"/api/collections/{cid}/documents").json()
    assert {d["source_path"] for d in docs} == {"handbook/leave.md", "handbook/hr/expenses.txt"}


def first_job_list(response) -> list[dict[str, Any]]:
    assert response.status_code == 202, response.text
    return response.json()["jobs"]


def test_unknown_ids_return_not_found(kb):
    for path in ("/api/documents/nope", "/api/documents/nope/file", "/api/jobs/nope", "/api/answers/nope",
                 "/api/collections/nope/stats", "/api/collections/nope/jobs"):
        assert_envelope(kb.client.get(path), 404, "not_found")


# ---------------------------------------------------------------- search


def test_search_returns_hits_scoped_to_the_collection(client):
    cid = make_collection(client)
    other = make_collection(client, "Other team")
    ingest(client, cid, [("leave.md", MARKDOWN), ("expenses.txt", TEXT)])
    ingest(client, other, [("other.txt", b"Parking spaces are assigned by the facilities desk every Monday.\n")])

    hits = client.post(f"/api/collections/{cid}/search", json={"query": "annual leave days", "top_k": 3}).json()
    assert 1 <= len(hits) <= 3
    assert all(h["chunk"]["collection_id"] == cid for h in hits)
    assert any("annual leave" in h["chunk"]["text"].lower() for h in hits)
    assert all(isinstance(h["score"], float) for h in hits)

    other_hits = client.post(f"/api/collections/{other}/search", json={"query": "annual leave days"}).json()
    assert all(h["chunk"]["collection_id"] == other for h in other_hits)

    assert_envelope(client.post(f"/api/collections/{cid}/search", json={"query": ""}), 422, "invalid_request")
    assert_envelope(client.post(f"/api/collections/{cid}/search", json={"query": "x", "top_k": 0}), 422,
                    "invalid_request")
    assert_envelope(client.post("/api/collections/nope/search", json={"query": "x"}), 404, "not_found")


# ---------------------------------------------------------------- ask, answers and feedback


def assert_answer_contract(body: dict[str, Any], cid: str, question: str) -> None:
    assert ANSWER_FIELDS <= set(body), ANSWER_FIELDS - set(body)
    assert body["status"] in {s.value for s in AnswerStatus}
    assert body["collection_id"] == cid and body["question"] == question
    assert isinstance(body["citations"], list)
    if body["status"] == "answered":
        assert body["answer"]
        markers = [c["marker"] for c in body["citations"]]
        assert markers == list(range(1, len(markers) + 1))
    else:
        assert body["answer"] is None
        assert body["message"]


def test_ask_returns_the_answer_contract(ask_client):
    client = ask_client
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN), ("expenses.txt", TEXT)])
    question = "How many days of annual leave do employees get?"
    response = client.post(f"/api/collections/{cid}/ask", json={"question": question, "top_k": 4})
    assert response.status_code == 200, response.text
    body = response.json()
    assert_answer_contract(body, cid, question)
    assert body["status"] in ("answered", "no_answer")
    assert body["knowledge_version"]

    stored = client.get(f"/api/answers/{body['id']}")
    assert stored.status_code == 200
    assert stored.json()["id"] == body["id"] and stored.json()["status"] == body["status"]


def test_ask_on_an_empty_collection_says_no_documents(ask_client):
    cid = make_collection(ask_client)
    body = ask_client.post(f"/api/collections/{cid}/ask", json={"question": "What is the leave policy?"}).json()
    assert_answer_contract(body, cid, "What is the leave policy?")
    assert body["status"] == "no_documents"


def test_ask_validates_input(ask_client):
    cid = make_collection(ask_client)
    url = f"/api/collections/{cid}/ask"
    assert_envelope(ask_client.post(url, json={"question": "  "}), 422, "invalid_request")
    assert_envelope(ask_client.post(url, json={"question": "q", "provider": "gpt"}), 422, "invalid_request")
    assert_envelope(ask_client.post(url, json={"question": "q", "model": "--dangerous"}), 422, "invalid_request")
    assert_envelope(ask_client.post(url, json={"question": "q", "top_k": 99}), 422, "invalid_request")
    assert_envelope(ask_client.post("/api/collections/nope/ask", json={"question": "q"}), 404, "not_found")


def test_provider_errors_become_answer_states(ask_client, monkeypatch):
    client = ask_client
    deps = client.app.state.deps
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    url = f"/api/collections/{cid}/ask"

    def not_signed_in(name=None, *, model=None):
        raise ProviderError("not_signed_in", "Not logged in", fix_command="claude /login")

    monkeypatch.setattr(deps.providers, "select", not_signed_in)
    body = client.post(url, json={"question": "Annual leave?", "provider": "claude_code"}).json()
    assert_answer_contract(body, cid, "Annual leave?")
    assert body["status"] == "provider_unavailable"
    assert "claude /login" in body["message"]

    def quota(name=None, *, model=None):
        raise ProviderError("rate_limited", "You have reached your usage limit.")

    monkeypatch.setattr(deps.providers, "select", quota)
    body = client.post(url, json={"question": "Annual leave?"}).json()
    assert body["status"] == "rate_limited" and body["answer"] is None


def test_an_unknown_model_is_refused_before_any_call(ask_client, monkeypatch):
    client = ask_client
    deps = client.app.state.deps
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    calls: list[str] = []

    class Claude:
        name = "claude_code"
        display_name = "Claude Code"

        def status(self):
            return ProviderStatus(name="claude_code", installed=True, signed_in=True, version="1.0", detail="",
                                  fix_command=None, models=["sonnet", "opus", "haiku"])

        def generate(self, system, prompt, *, timeout_s, model=None):
            calls.append(model or "")
            raise AssertionError("an unknown model must be refused before any call")

    monkeypatch.setitem(deps.providers._providers, "claude_code", Claude())
    deps.providers.invalidate()
    body = client.post(f"/api/collections/{cid}/ask",
                       json={"question": "Annual leave?", "provider": "claude_code", "model": "gpt-6-sol"}).json()
    assert_answer_contract(body, cid, "Annual leave?")
    assert body["status"] == "provider_unavailable"
    assert body["message"] == "Claude Code cannot use the model gpt-6-sol. Choose one of: sonnet, opus, haiku."
    assert calls == []


def test_search_failure_becomes_retrieval_unavailable(ask_client, monkeypatch):
    RetrievalUnavailable = pytest.importorskip("ragkb.retrieval.service").RetrievalUnavailable
    client = ask_client
    deps = client.app.state.deps
    cid = make_collection(client)

    def down(*args, **kwargs):
        raise RetrievalUnavailable("store offline")

    monkeypatch.setattr(deps.retriever, "search", down)
    body = client.post(f"/api/collections/{cid}/ask", json={"question": "Anything?"}).json()
    assert_answer_contract(body, cid, "Anything?")
    assert body["status"] == "retrieval_unavailable"
    assert_envelope(client.post(f"/api/collections/{cid}/search", json={"query": "x"}), 503,
                    "retrieval_unavailable")


def test_audit_never_records_the_question_or_answer_text(ask_client, monkeypatch):
    client = ask_client
    deps = client.app.state.deps
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    rows: list[tuple[str, dict[str, Any]]] = []
    original = deps.meta.audit

    def spy(event: str, **fields: Any) -> None:
        rows.append((event, fields))
        original(event, **fields)

    monkeypatch.setattr(deps.meta, "audit", spy)
    question = "Which note is needed for sick leave?"
    body = client.post(f"/api/collections/{cid}/ask", json={"question": question}).json()
    events = [fields for event, fields in rows if event == "answer"]
    assert len(events) == 1
    fields = events[0]
    assert fields["question_chars"] == len(question)
    assert {"provider", "model", "status", "elapsed_ms"} <= set(fields)
    flat = " ".join(str(v) for v in fields.values())
    assert question not in flat
    if body["answer"]:
        assert body["answer"][:40] not in flat


def test_feedback_on_a_real_answer(ask_client):
    client = ask_client
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    answer = client.post(f"/api/collections/{cid}/ask", json={"question": "Annual leave?"}).json()
    url = f"/api/answers/{answer['id']}/feedback"

    response = client.post(url, json={"rating": "up", "note": "Clear and correct"})
    assert response.status_code == 201
    assert response.json()["rating"] == "up" and response.json()["answer_id"] == answer["id"]
    assert client.post(url, json={"rating": "down"}).status_code == 201
    read_back = client.get(url).json()
    assert [f["rating"] for f in read_back] == ["down", "up"]
    assert read_back[0]["question"] == "Annual leave?"


def test_ask_rate_limit(ask_client):
    client = ask_client
    client.app.state.deps.ask_limiter = RateLimiter(2, 60)
    cid = make_collection(client)
    url = f"/api/collections/{cid}/ask"
    for _ in range(2):
        assert client.post(url, json={"question": "Annual leave?"}).status_code == 200
    limited = client.post(url, json={"question": "Annual leave?"})
    assert_envelope(limited, 429, "rate_limited")
    assert int(limited.headers["retry-after"]) >= 1
    other = make_collection(client, "Second")
    assert client.post(f"/api/collections/{other}/ask", json={"question": "Annual leave?"}).status_code == 200


def test_providers_list_includes_extractive(ask_client):
    response = ask_client.get("/api/providers")
    assert response.status_code == 200
    statuses = response.json()
    names = {s["name"] for s in statuses}
    assert "extractive" in names
    extractive = next(s for s in statuses if s["name"] == "extractive")
    assert extractive["installed"] and extractive["signed_in"]
    assert ask_client.get("/api/providers?refresh=1").status_code == 200


def test_health_reports_counts(client):
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    body = client.get("/api/health").json()
    assert body["counts"]["collections"] == 1 and body["counts"]["documents"] == 1
    assert body["counts"]["documents_ready"] == 1 and body["counts"]["documents_total"] == 1
    assert body["counts"]["chunks"] >= 1
    assert body["store"] == {"metadata": "ok", "vector": "ok"}
    assert body["knowledge_version"]
    assert ".md" in body["supported_types"]


def test_mcp_is_mounted_when_enabled(settings):
    _require(*CORE_PARTS, *ANSWER_PARTS, "ragkb.mcp.server")
    app = create_app(settings.model_copy(update={"MCP_ENABLED": True}))
    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        mcp = client.get("/api/health").json()["mcp"]
        assert mcp["enabled"] and mcp["mounted"]
        assert "ask_knowledge_base" in mcp["tools"]
