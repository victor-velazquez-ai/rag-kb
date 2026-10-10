"""MCP server: the tools over streamable HTTP with the real mcp client, error shapes, the loopback guard.

The index tests build a temporary collection through the real ingestion pipeline with FakeEmbedder.
Everything skips cleanly while the parts it needs are not installed.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import httpx2
import pytest
from fastapi import FastAPI
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from ragkb.config import Settings
from ragkb.models import Collection, Document, Hit

server = pytest.importorskip("ragkb.mcp.server")
service = pytest.importorskip("ragkb.retrieval.service")

LOCAL_BASE = "http://127.0.0.1:8765"
TOOL_NAMES = {"list_collections", "list_documents", "search_knowledge_base", "ask_knowledge_base"}

HANDBOOK = """# Employee handbook

## Annual leave
Every employee receives 27 days of paid annual leave per calendar year, plus public holidays.
Unused leave of up to five days can be carried over to the next year with manager approval.

## Working hours
Core working hours are 10:00 to 16:00. Flexible time is recorded in the time tracking tool.
"""
SECURITY = """# IT security policy

## Passwords
Passwords must have at least 14 characters. Use the company password manager for every account.
Multi-factor authentication is mandatory for email, the VPN and the source code host.

## Lost devices
Report a lost or stolen laptop to the service desk within one hour, by phone at extension 4400.
"""
TRAVEL = """# Travel and expense policy

## Meals
The daily meal allowance on business trips is 45 euros. Alcohol is never reimbursed.

## Booking
Book flights through the travel portal at least 14 days before departure. Economy class only.
"""


# ---------------------------------------------------------------- helpers


def run(fn: Callable[[], Any]) -> Any:
    return anyio.run(fn)


@asynccontextmanager
async def mcp_client(app: FastAPI, base_url: str = LOCAL_BASE, *, mode: str = "auto") -> AsyncIterator[Client]:
    """Run the app's lifespan and connect the real mcp client through an in-process ASGI transport."""
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=base_url)
        async with http:
            transport = streamable_http_client(f"{base_url}/mcp", http_client=http)
            async with Client(transport, mode=mode) as client:
                yield client


def make_app(deps: Any) -> FastAPI:
    app = FastAPI()
    server.mount_mcp(app, deps)
    return app


def payload_of(result: Any) -> dict[str, Any]:
    """The structured content, checked against the text content (they must say the same thing)."""
    assert result.content, "a tool result must carry text content"
    text = result.content[0].text
    assert "Traceback" not in text
    assert json.loads(text) == result.structured_content
    return result.structured_content


def assert_error(result: Any, code: str) -> dict[str, Any]:
    assert result.is_error is True
    payload = payload_of(result)
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message"}
    assert payload["error"]["code"] == code
    assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]
    return payload["error"]


def call(app: FastAPI, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    async def go() -> Any:
        async with mcp_client(app, **kwargs) as client:
            return await client.call_tool(name, arguments or {})

    return run(go)


# ---------------------------------------------------------------- a tiny fake for the error paths


NOW = datetime.now(UTC)


class FakeMeta:
    def __init__(self, *, broken: bool = False) -> None:
        self.broken = broken
        self.collection = Collection(id="c1", name="Fake", created_at=NOW)
        self.audits: list[tuple[str, dict[str, Any]]] = []
        self.answers: list[str] = []

    def get_collection(self, collection_id: str) -> Collection | None:
        return self.collection if collection_id == "c1" else None

    def list_collections(self) -> list[Collection]:
        if self.broken:
            raise RuntimeError("disk failure at C:\\secret\\meta.sqlite")
        return [self.collection]

    def list_documents(self, collection_id: str) -> list[Document]:
        return []

    def current_version(self) -> str:
        return "v20260101.1"

    def audit(self, event: str, **fields: Any) -> None:
        self.audits.append((event, fields))

    def save_answer(self, answer_json: str, answer_id: str, collection_id: str) -> None:
        self.answers.append(answer_id)


class DownRetriever:
    def search(self, collection_id: str, query: str, top_k: int) -> list[Hit]:
        raise service.RetrievalUnavailable("the vector store is down")


class FakeStore:
    def has_documents(self, collection_id: str) -> bool:
        return True


def fake_deps(*, public: bool = False, broken: bool = False, retriever: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        settings=Settings(_env_file=None, MCP_PUBLIC=public, DATA_DIR=Path("unused")),
        meta=FakeMeta(broken=broken),
        store=FakeStore(),
        embedder=None,
        retriever=retriever or DownRetriever(),
        pipeline=None,
        providers=None,
    )


# ---------------------------------------------------------------- a real index


@pytest.fixture
def kb(tmp_path: Path) -> SimpleNamespace:
    sqlite_meta = pytest.importorskip("ragkb.store.sqlite_meta")
    lancedb_store = pytest.importorskip("ragkb.store.lancedb_store")
    embedder_mod = pytest.importorskip("ragkb.retrieval.embedder")
    pipeline_mod = pytest.importorskip("ragkb.ingest.pipeline")

    settings = Settings(_env_file=None, DATA_DIR=tmp_path / "data")
    settings.ensure_dirs()
    meta = sqlite_meta.SQLiteMeta(settings.meta_path)
    embedder = embedder_mod.FakeEmbedder(dims=384)
    store = lancedb_store.LanceDBStore(settings.lancedb_dir, embedder.dims)
    retriever = service.Retriever(store, embedder, reranker=None, min_score=0.0)
    pipeline = pipeline_mod.IngestPipeline(settings, meta, store, embedder)

    collection = meta.create_collection("Sample", "three short policies")
    empty = meta.create_collection("Empty")
    source = tmp_path / "docs"
    source.mkdir()
    for name, text in (("handbook.md", HANDBOOK), ("security.md", SECURITY), ("travel.md", TRAVEL)):
        path = source / name
        path.write_text(text, encoding="utf-8")
        pipeline.ingest_file(collection.id, path)

    deps = SimpleNamespace(
        settings=settings, meta=meta, store=store, embedder=embedder, retriever=retriever,
        pipeline=pipeline, providers=None,
    )
    yield SimpleNamespace(deps=deps, app=make_app(deps), collection=collection, empty=empty)
    for component in (store, meta):
        closer = getattr(component, "close", None)
        if callable(closer):
            closer()


# ---------------------------------------------------------------- tools list


def test_tools_list_names_and_schemas() -> None:
    app = make_app(fake_deps())

    async def go() -> Any:
        async with mcp_client(app) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in run(go).tools}
    assert set(tools) == TOOL_NAMES
    for tool in tools.values():
        assert tool.description
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
        assert tool.annotations is not None and tool.annotations.read_only_hint is True
    for name in ("search_knowledge_base", "ask_knowledge_base"):
        top_k = tools[name].input_schema["properties"]["top_k"]
        assert (top_k["minimum"], top_k["maximum"], top_k["default"]) == (1, 20, 6)
    assert tools["ask_knowledge_base"].input_schema["required"] == ["collection_id", "question"]


def test_legacy_handshake_clients_work_too() -> None:
    app = make_app(fake_deps())
    result = call(app, "list_collections", mode="legacy")
    assert result.is_error is False
    assert payload_of(result)["collections"][0]["id"] == "c1"


# ---------------------------------------------------------------- each tool on a real index


def test_list_collections(kb: SimpleNamespace) -> None:
    payload = payload_of(call(kb.app, "list_collections"))
    by_name = {c["name"]: c for c in payload["collections"]}
    assert by_name["Sample"]["documents"] == 3
    assert by_name["Sample"]["id"] == kb.collection.id
    assert by_name["Empty"]["documents"] == 0
    assert payload["knowledge_version"].startswith("v")


def test_list_documents_by_id_and_by_name(kb: SimpleNamespace) -> None:
    payload = payload_of(call(kb.app, "list_documents", {"collection_id": kb.collection.id}))
    assert payload["collection_id"] == kb.collection.id
    docs = payload["documents"]
    assert sorted(d["filename"] for d in docs) == ["handbook.md", "security.md", "travel.md"]
    assert all(d["status"] == "ready" and d["chunks"] > 0 and d["indexed_at"] for d in docs)
    by_name = payload_of(call(kb.app, "list_documents", {"collection_id": "sample"}))
    assert by_name["collection_id"] == kb.collection.id


def test_search_returns_ranked_passages(kb: SimpleNamespace) -> None:
    result = call(kb.app, "search_knowledge_base",
                  {"collection_id": kb.collection.id, "query": "meal allowance on business trips", "top_k": 3})
    assert result.is_error is False
    payload = payload_of(result)
    assert payload["status"] == "ok"
    results = payload["results"]
    assert 1 <= len(results) <= 3
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    for row in results:
        assert {"chunk_id", "doc_id", "title", "section", "page", "text", "score"} <= set(row)
        assert row["text"].strip()
    assert any("45 euros" in r["text"] for r in results)


def test_ask_returns_a_brief_with_guidance(kb: SimpleNamespace) -> None:
    result = call(kb.app, "ask_knowledge_base",
                  {"collection_id": kb.collection.id, "question": "How many days of annual leave do employees get?"})
    assert result.is_error is False
    brief = payload_of(result)
    for key in ("id", "question", "collection_id", "status", "answer", "message", "citations",
                "passages_considered", "provider", "knowledge_version", "language", "created_at",
                "passages", "guidance"):
        assert key in brief
    assert brief["provider"]["name"] == "extractive"
    assert "[n]" in brief["guidance"] and "language" in brief["guidance"]
    assert brief["status"] == "answered"
    markers = [c["marker"] for c in brief["citations"]]
    assert markers == list(range(1, len(markers) + 1)) and markers
    assert [p["marker"] for p in brief["passages"]] == markers
    assert any("27 days" in p["text"] for p in brief["passages"])


def test_ask_on_an_empty_collection_is_not_an_answer(kb: SimpleNamespace) -> None:
    brief = payload_of(call(kb.app, "ask_knowledge_base", {"collection_id": kb.empty.id, "question": "Anything?"}))
    assert brief["status"] == "no_documents"
    assert brief["answer"] is None
    assert brief["citations"] == [] and brief["passages"] == []
    assert brief["guidance"]


def test_search_on_an_empty_collection_says_so(kb: SimpleNamespace) -> None:
    payload = payload_of(call(kb.app, "search_knowledge_base", {"collection_id": kb.empty.id, "query": "leave"}))
    assert payload["results"] == []
    assert payload["status"] == "no_documents"


# ---------------------------------------------------------------- error shapes


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("list_documents", {"collection_id": "missing"}, "not_found"),
        ("search_knowledge_base", {"collection_id": "missing", "query": "x"}, "not_found"),
        ("ask_knowledge_base", {"collection_id": "missing", "question": "x"}, "not_found"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "   "}, "invalid_request"),
        ("ask_knowledge_base", {"collection_id": "c1", "question": ""}, "invalid_request"),
        ("ask_knowledge_base", {"collection_id": "c1"}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x", "top_k": 0}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x", "top_k": 21}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x", "top_k": "many"}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x", "top_k": True}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x" * 2001}, "invalid_request"),
        ("search_knowledge_base", {"collection_id": "c1", "query": "x", "limit": 3}, "invalid_request"),
        ("list_documents", {"collection_id": 42}, "invalid_request"),
        ("delete_everything", {}, "unknown_tool"),
    ],
)
def test_tool_errors_are_structured(name: str, arguments: dict[str, Any], code: str) -> None:
    app = make_app(fake_deps())
    assert_error(call(app, name, arguments), code)


def test_unexpected_failures_do_not_leak_details() -> None:
    app = make_app(fake_deps(broken=True))
    error = assert_error(call(app, "list_collections"), "internal_error")
    assert "secret" not in error["message"] and "RuntimeError" not in error["message"]


def test_search_unavailable_is_an_error_and_ask_is_a_status() -> None:
    app = make_app(fake_deps())
    assert_error(call(app, "search_knowledge_base", {"collection_id": "c1", "query": "leave"}), "retrieval_unavailable")
    result = call(app, "ask_knowledge_base", {"collection_id": "c1", "question": "leave?"})
    assert result.is_error is False
    brief = payload_of(result)
    assert brief["status"] == "retrieval_unavailable"
    assert brief["answer"] is None and brief["message"]
    assert brief["guidance"]


def test_top_k_accepts_whole_numbers_sent_as_text() -> None:
    tools = server.KnowledgeBaseTools(fake_deps())
    payload, is_error = tools.call("search_knowledge_base", {"collection_id": "c1", "query": "x", "top_k": "3"})
    assert is_error is True and payload["error"]["code"] == "retrieval_unavailable"


# ---------------------------------------------------------------- loopback guard


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1:8765", True),
        ("127.0.0.1", True),
        ("localhost:8765", True),
        ("LOCALHOST", True),
        ("[::1]:8765", True),
        ("127.4.5.6:80", True),
        ("kb.example.com", False),
        ("192.168.1.20:8765", False),
        ("localhost.example.com", False),
        ("[::1", False),
        ("", False),
        (None, False),
    ],
)
def test_host_is_loopback(host: str | None, expected: bool) -> None:
    assert server.host_is_loopback(host) is expected


def test_origin_is_loopback() -> None:
    assert server.origin_is_loopback(None) is True
    assert server.origin_is_loopback("http://localhost:8765") is True
    assert server.origin_is_loopback("https://evil.example.com") is False


def _raw_post(app: FastAPI, headers: dict[str, str]) -> httpx2.Response:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

    async def go() -> httpx2.Response:
        transport = httpx2.ASGITransport(app=app)
        async with app.router.lifespan_context(app), httpx2.AsyncClient(transport=transport, base_url=LOCAL_BASE) as http:
            return await http.post("/mcp", json=body, headers={"accept": "application/json, text/event-stream", **headers})

    return run(go)


def test_non_loopback_host_is_refused() -> None:
    response = _raw_post(make_app(fake_deps()), {"host": "kb.example.com"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert "RAGKB_MCP_PUBLIC" in response.json()["error"]["message"]


def test_foreign_origin_is_refused() -> None:
    response = _raw_post(make_app(fake_deps()), {"origin": "https://evil.example.com"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_public_flag_lets_other_hosts_in() -> None:
    app = make_app(fake_deps(public=True))
    result = call(app, "list_collections", base_url="http://kb.example.com")
    assert result.is_error is False


def test_requests_before_startup_get_a_plain_error() -> None:
    app = make_app(fake_deps())

    async def go() -> httpx2.Response:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=LOCAL_BASE) as http:
            return await http.post("/mcp", json={}, headers={"accept": "application/json"})

    response = run(go)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"


def test_disabled_mcp_mounts_nothing() -> None:
    deps = fake_deps()
    deps.settings = Settings(_env_file=None, MCP_ENABLED=False, DATA_DIR=Path("unused"))
    app = make_app(deps)
    assert not any(getattr(route, "path", None) == "/mcp" for route in app.router.routes)


def test_route_wins_over_a_catch_all_mount() -> None:
    staticfiles = pytest.importorskip("starlette.staticfiles")
    app = FastAPI()
    app.mount("/", staticfiles.StaticFiles(directory=str(Path(__file__).parent)), name="everything")
    server.mount_mcp(app, fake_deps())
    result = call(app, "list_collections")
    assert result.is_error is False


def test_mounted_on_the_full_app(tmp_path: Path) -> None:
    app_mod = pytest.importorskip("ragkb.server.app")
    deps_mod = pytest.importorskip("ragkb.server.deps")
    settings = Settings(_env_file=None, DATA_DIR=tmp_path / "data")
    deps = deps_mod.build_deps(settings, fake_embedder=True)
    if deps.meta is None:
        pytest.skip("the metadata store part is not installed yet")
    app = app_mod.create_app(settings, deps=deps)
    assert deps.mcp_mounted is True
    payload = payload_of(call(app, "list_collections"))
    assert payload["collections"] == []
