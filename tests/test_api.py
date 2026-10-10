"""HTTP API tests. Offline: fake embedder, extractive provider, a temporary data folder.

Tests that need the ingestion, store and retrieval part or the providers and answer part skip
cleanly while those parts are not installed.
"""
from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ragkb.config import Settings
from ragkb.models import Answer, AnswerStatus
from ragkb.providers.base import ProviderError
from ragkb.server.app import create_app
from ragkb.server.deps import RateLimiter, build_deps

CORE_PARTS = (
    "ragkb.store.sqlite_meta",
    "ragkb.store.lancedb_store",
    "ragkb.retrieval.embedder",
    "ragkb.retrieval.service",
    "ragkb.ingest.parsers",
    "ragkb.ingest.pipeline",
)
ANSWER_PARTS = ("ragkb.providers.registry", "ragkb.answer.compose")

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


def _require(*modules: str) -> None:
    for name in modules:
        pytest.importorskip(name)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("RAGKB_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("RAGKB_PROVIDER", "extractive")
    return Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        PROVIDER="extractive",
        MCP_ENABLED=False,
        MAX_UPLOAD_MB=1,
        LOG_LEVEL="WARNING",
    )


@pytest.fixture
def bare_client(settings):
    """An app with whatever parts are installed; works with none of them."""
    app = create_app(settings)
    with TestClient(app, base_url=LOCAL_BASE, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client(settings):
    _require(*CORE_PARTS)
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


def upload(client: TestClient, cid: str, files: list[tuple[str, bytes]], title: str | None = None):
    parts = [("files", (name, data, "application/octet-stream")) for name, data in files]
    return client.post(f"/api/collections/{cid}/documents", files=parts, data={"title": title} if title else None)


def wait_for_job(client: TestClient, job_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("ready", "failed"):
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job did not finish: {job}")
        time.sleep(0.05)


def ingest(client: TestClient, cid: str, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    response = upload(client, cid, files)
    assert response.status_code == 202, response.text
    return [wait_for_job(client, job["id"]) for job in response.json()["jobs"]]


# ---------------------------------------------------------------- envelope and basics (no parts needed)


def test_health_answers_even_without_parts(bare_client):
    body = bare_client.get("/api/health").json()
    assert body["status"] in ("ok", "degraded")
    for key in ("version", "knowledge_version", "counts", "store", "data_dir", "mcp", "max_upload_mb"):
        assert key in body


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
    with TestClient(create_app(settings), base_url="http://kb.example.com", raise_server_exceptions=False) as remote:
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
    with TestClient(public, base_url="http://kb.example.com", raise_server_exceptions=False) as remote:
        assert_envelope(remote.get("/api/does-not-exist"), 404, "not_found")
        assert remote.get("/api/health").json()["api"] == {"public": True}


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


# ---------------------------------------------------------------- collections


def test_collections_crud(client):
    cid = make_collection(client)
    listed = client.get("/api/collections").json()
    assert [c["id"] for c in listed] == [cid]
    assert listed[0]["name"] == "Handbook" and listed[0]["documents"] == 0

    assert_envelope(client.post("/api/collections", json={"name": "handbook"}), 409, "conflict")
    assert_envelope(client.post("/api/collections", json={"name": "   "}), 422, "invalid_request")

    stats = client.get(f"/api/collections/{cid}/stats").json()
    assert stats["documents"] == 0 and stats["chunks"] == 0

    deleted = client.delete(f"/api/collections/{cid}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] == cid
    assert client.get("/api/collections").json() == []
    assert_envelope(client.get(f"/api/collections/{cid}/documents"), 404, "not_found")
    assert_envelope(client.delete(f"/api/collections/{cid}"), 404, "not_found")


def test_deleting_a_collection_removes_its_documents(client):
    cid = make_collection(client)
    jobs = ingest(client, cid, [("leave.md", MARKDOWN)])
    doc_id = jobs[0]["doc_id"]
    assert client.delete(f"/api/collections/{cid}").status_code == 200
    assert_envelope(client.get(f"/api/documents/{doc_id}"), 404, "not_found")


# ---------------------------------------------------------------- upload and documents


def test_upload_markdown_and_text_reach_ready(client):
    cid = make_collection(client)
    response = upload(client, cid, [("leave-policy.md", MARKDOWN), ("expenses.txt", TEXT)])
    assert response.status_code == 202, response.text
    jobs = response.json()["jobs"]
    assert len(jobs) == 2
    assert all(j["status"] == "queued" and j["already_indexed"] is False for j in jobs)
    assert [j["filename"] for j in jobs] == ["leave-policy.md", "expenses.txt"]

    finished = [wait_for_job(client, j["id"]) for j in jobs]
    assert [j["status"] for j in finished] == ["ready", "ready"], finished
    assert all(j["doc_id"] for j in finished)
    assert all(j["finished_at"] for j in finished)

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
    assert stats["by_status"] == {"ready": 2}
    assert client.get("/api/collections").json()[0]["documents"] == 2


def test_upload_job_keeps_its_document_after_a_restart(settings):
    _require(*CORE_PARTS)
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


def test_unsupported_type_is_refused(client):
    cid = make_collection(client)
    assert_envelope(upload(client, cid, [("tool.exe", b"MZ binary")]), 415, "unsupported_file_type")
    assert_envelope(upload(client, cid, [("leave.md", MARKDOWN), ("run.sh", b"echo")]), 415,
                    "unsupported_file_type")
    assert client.get(f"/api/collections/{cid}/documents").json() == []


def test_size_cap_is_enforced(client):
    cid = make_collection(client)
    too_big = b"word " * (1024 * 1024 // 5 + 10)
    assert len(too_big) > 1024 * 1024
    assert_envelope(upload(client, cid, [("big.txt", too_big)]), 413, "file_too_large")


def test_empty_file_and_title_rules(client):
    cid = make_collection(client)
    assert_envelope(upload(client, cid, [("empty.txt", b"")]), 400, "empty_document")
    assert_envelope(upload(client, cid, [("a.md", MARKDOWN), ("b.txt", TEXT)], title="One title"), 400,
                    "invalid_request")
    assert_envelope(upload(client, "no-such-collection", [("a.md", MARKDOWN)]), 404, "not_found")


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


def test_unknown_ids_return_not_found(client):
    for path in ("/api/documents/nope", "/api/documents/nope/file", "/api/jobs/nope", "/api/answers/nope",
                 "/api/collections/nope/stats", "/api/collections/nope/jobs"):
        assert_envelope(client.get(path), 404, "not_found")


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

    def not_signed_in(name=None):
        raise ProviderError("not_signed_in", "Not logged in", fix_command="claude /login")

    monkeypatch.setattr(deps.providers, "select", not_signed_in)
    body = client.post(url, json={"question": "Annual leave?", "provider": "claude_code"}).json()
    assert_answer_contract(body, cid, "Annual leave?")
    assert body["status"] == "provider_unavailable"
    assert "claude /login" in body["message"]

    def quota(name=None):
        raise ProviderError("rate_limited", "You have reached your usage limit.")

    monkeypatch.setattr(deps.providers, "select", quota)
    body = client.post(url, json={"question": "Annual leave?"}).json()
    assert body["status"] == "rate_limited" and body["answer"] is None


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


def test_feedback(ask_client):
    client = ask_client
    cid = make_collection(client)
    ingest(client, cid, [("leave.md", MARKDOWN)])
    answer = client.post(f"/api/collections/{cid}/ask", json={"question": "Annual leave?"}).json()
    url = f"/api/answers/{answer['id']}/feedback"

    response = client.post(url, json={"rating": "up", "note": "Clear and correct"})
    assert response.status_code == 201
    assert response.json()["rating"] == "up" and response.json()["answer_id"] == answer["id"]
    assert client.post(url, json={"rating": "down"}).status_code == 201
    assert_envelope(client.post(url, json={"rating": "meh"}), 422, "invalid_request")
    assert_envelope(client.post("/api/answers/nope/feedback", json={"rating": "up"}), 404, "not_found")


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
