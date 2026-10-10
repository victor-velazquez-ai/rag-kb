"""MCP server: the knowledge base as tools that an assistant (Claude Desktop, claude.ai, Claude Code,
Codex, ChatGPT) can call, over streamable HTTP at /mcp.

The tools never call a model. `ask_knowledge_base` returns the answer contract produced with the
extractive provider plus a short `guidance` text, so the calling assistant writes the final answer on
its user's own subscription.

Every tool error comes back as `{"error": {"code", "message"}}` content with `isError` set, never as a
stack trace. Unless `RAGKB_MCP_PUBLIC=true`, requests whose Host (or Origin) is not loopback are refused.

Call `mount_mcp(app, deps)` while building the app, before it starts: it registers the `/mcp` route
first in the routing table and ties the MCP session manager to the app's lifespan.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

import anyio
from fastapi import FastAPI
from mcp import types
from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from ragkb import __version__
from ragkb.answer.compose import compose, compose_unavailable
from ragkb.models import Collection, Hit
from ragkb.providers.extractive import ExtractiveProvider
from ragkb.retrieval.service import RetrievalUnavailable
from ragkb.server.loopback import host_is_loopback, origin_is_loopback

if TYPE_CHECKING:
    from ragkb.server.deps import Deps

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"
SERVER_NAME = "rag-kb"
TOP_K_DEFAULT = 6
TOP_K_MIN = 1
TOP_K_MAX = 20
MAX_QUERY_CHARS = 2000

GUIDANCE = (
    "Write the final answer yourself from this brief, in the language the user wrote in, starting with "
    "the direct answer in one or two sentences and then the useful details. Cite inline with the [n] "
    "markers exactly as they appear in `citations` and `passages`, right after the sentence each one "
    "supports, and never cite a number that is not listed there. If the status is not `answered`, or "
    "the brief does not cover the question, say plainly that the knowledge base does not cover it "
    "instead of guessing or filling the gap from general knowledge."
)

INSTRUCTIONS = (
    "rag-kb is a private document knowledge base. Call list_collections first to find a collection id. "
    "Use ask_knowledge_base to answer a question from the documents: it returns a brief (status, an "
    "extractive digest, numbered citations and passages) and guidance on writing the final answer. "
    "Use search_knowledge_base when you only need the raw passages, and list_documents to see what a "
    "collection holds. Treat a status other than `answered` as 'the documents do not cover this'."
)

_COLLECTION_ID_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "description": "The collection id from list_collections (an exact collection name also works).",
}
_TOP_K_SCHEMA = {
    "type": "integer",
    "minimum": TOP_K_MIN,
    "maximum": TOP_K_MAX,
    "default": TOP_K_DEFAULT,
    "description": f"How many passages to retrieve, {TOP_K_MIN} to {TOP_K_MAX}.",
}
_READ_ONLY = types.ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)

TOOLS: list[types.Tool] = [
    types.Tool(
        name="list_collections",
        title="List collections",
        description=(
            "List the knowledge base collections with their id, name, description and number of "
            "searchable documents. Call this first to get a collection id."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations=_READ_ONLY,
    ),
    types.Tool(
        name="list_documents",
        title="List documents",
        description=(
            "List the documents in one collection: id, title, file name, status, pages, passages "
            "(chunks) and when each was indexed."
        ),
        input_schema={
            "type": "object",
            "properties": {"collection_id": _COLLECTION_ID_SCHEMA},
            "required": ["collection_id"],
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    ),
    types.Tool(
        name="search_knowledge_base",
        title="Search the knowledge base",
        description=(
            "Hybrid search (meaning plus keywords) over one collection. Returns the best passages in "
            "rank order with title, section, page, text and score. No answer is written."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "collection_id": _COLLECTION_ID_SCHEMA,
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                    "description": "What to look for, in natural language or keywords.",
                },
                "top_k": _TOP_K_SCHEMA,
            },
            "required": ["collection_id", "query"],
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    ),
    types.Tool(
        name="ask_knowledge_base",
        title="Ask the knowledge base",
        description=(
            "Answer a question from one collection's documents. Returns a brief: a status (answered, "
            "no_answer, no_documents, retrieval_unavailable), an extractive digest with [n] markers, "
            "numbered citations, the cited passages in full, and guidance. Write the final answer from "
            "the brief, citing with the same [n] markers; when the status is not answered, tell the "
            "user the documents do not cover the question."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "collection_id": _COLLECTION_ID_SCHEMA,
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                    "description": "The user's question, in the user's own words and language.",
                },
                "top_k": _TOP_K_SCHEMA,
            },
            "required": ["collection_id", "question"],
            "additionalProperties": False,
        },
        annotations=_READ_ONLY,
    ),
]
_TOOLS_BY_NAME: dict[str, types.Tool] = {tool.name: tool for tool in TOOLS}


class ToolFailure(Exception):
    """A tool error that is returned to the caller as `{"error": {"code", "message"}}`."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def payload(self) -> dict[str, Any]:
        return error_payload(self.code, self.message)


def error_payload(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


# ---------------------------------------------------------------- argument validation


def _check_arguments(tool: str, arguments: Mapping[str, Any]) -> None:
    schema = _TOOLS_BY_NAME[tool].input_schema
    allowed = set(schema.get("properties", {}))
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        accepted = ", ".join(sorted(allowed)) or "none"
        raise ToolFailure(
            "invalid_request",
            f"Unknown argument(s) for {tool}: {', '.join(unknown)}. Accepted arguments: {accepted}.",
        )
    missing = [name for name in schema.get("required", []) if arguments.get(name) is None]
    if missing:
        raise ToolFailure("invalid_request", f"Missing required argument(s): {', '.join(missing)}.")


def _text_argument(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolFailure("invalid_request", f"`{name}` must be a non-empty string.")
    value = value.strip()
    if len(value) > MAX_QUERY_CHARS:
        raise ToolFailure("invalid_request", f"`{name}` is longer than {MAX_QUERY_CHARS} characters.")
    return value


def _top_k_argument(arguments: Mapping[str, Any]) -> int:
    value = arguments.get("top_k")
    if value is None:
        return TOP_K_DEFAULT
    number: int | None = None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    if number is None or not TOP_K_MIN <= number <= TOP_K_MAX:
        raise ToolFailure("invalid_request", f"`top_k` must be a whole number from {TOP_K_MIN} to {TOP_K_MAX}.")
    return number


# ---------------------------------------------------------------- the tools


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _status_value(value: Any) -> str:
    return getattr(value, "value", value)


class KnowledgeBaseTools:
    """The four tools as plain synchronous methods over the shared dependencies.

    `call()` validates the arguments and returns `(payload, is_error)`; it never raises.
    """

    def __init__(self, deps: Deps):
        self._deps = deps
        self._extractive = ExtractiveProvider()

    # ---- dispatch

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
        started = time.perf_counter()
        try:
            if name not in _TOOLS_BY_NAME:
                raise ToolFailure("unknown_tool", f"There is no tool named {name!r}.")
            if arguments is not None and not isinstance(arguments, Mapping):
                raise ToolFailure("invalid_request", "Arguments must be a JSON object.")
            args: Mapping[str, Any] = arguments or {}
            _check_arguments(name, args)
            if name == "list_collections":
                payload = self.list_collections()
            elif name == "list_documents":
                payload = self.list_documents(_text_argument(args, "collection_id"))
            elif name == "search_knowledge_base":
                payload = self.search(
                    _text_argument(args, "collection_id"), _text_argument(args, "query"), _top_k_argument(args)
                )
            else:
                payload = self.ask(
                    _text_argument(args, "collection_id"), _text_argument(args, "question"), _top_k_argument(args)
                )
            return payload, False
        except ToolFailure as failure:
            logger.info("mcp tool %s refused: %s", name, failure.code)
            return failure.payload(), True
        except Exception:
            logger.exception("mcp tool %s failed", name)
            return error_payload("internal_error", "The knowledge base hit an unexpected error. See the server log."), True
        finally:
            logger.debug("mcp tool %s took %d ms", name, int((time.perf_counter() - started) * 1000))

    # ---- helpers

    def _component(self, name: str, label: str) -> Any:
        value = getattr(self._deps, name, None)
        if value is None:
            raise ToolFailure("unavailable", f"The {label} is not available right now.")
        return value

    @property
    def _meta(self) -> Any:
        return self._component("meta", "metadata store")

    def _resolve_collection(self, value: str) -> Collection:
        meta = self._meta
        collection = meta.get_collection(value)
        if collection is not None:
            return collection
        wanted = value.casefold()
        matches = [c for c in meta.list_collections() if c.name.casefold() == wanted]
        if len(matches) == 1:
            return matches[0]
        raise ToolFailure("not_found", "No collection has that id. Call list_collections to see the available ids.")

    def _version(self) -> str:
        try:
            return str(self._meta.current_version())
        except ToolFailure:
            raise
        except Exception:
            logger.warning("could not read the knowledge version", exc_info=True)
            return ""

    def _has_documents(self, collection_id: str) -> bool:
        store = getattr(self._deps, "store", None)
        if store is None:
            return False
        try:
            return bool(store.has_documents(collection_id))
        except Exception:
            logger.warning("could not check whether the collection has documents", exc_info=True)
            return False

    def _audit(self, event: str, **fields: Any) -> None:
        try:
            self._meta.audit(event, **fields)
        except Exception:
            logger.warning("could not write the %s audit event", event, exc_info=True)

    # ---- tools

    def list_collections(self) -> dict[str, Any]:
        meta = self._meta
        rows = []
        for collection in meta.list_collections():
            documents = meta.list_documents(collection.id)
            ready = sum(1 for d in documents if _status_value(d.status) == "ready")
            rows.append(
                {
                    "id": collection.id,
                    "name": collection.name,
                    "description": collection.description,
                    "documents": ready,
                }
            )
        return {"collections": rows, "knowledge_version": self._version()}

    def list_documents(self, collection_id: str) -> dict[str, Any]:
        collection = self._resolve_collection(collection_id)
        documents = [
            {
                "id": d.id,
                "title": d.title,
                "filename": d.filename,
                "status": _status_value(d.status),
                "pages": d.pages,
                "chunks": d.chunk_count,
                "indexed_at": _iso(d.indexed_at),
            }
            for d in self._meta.list_documents(collection.id)
        ]
        return {
            "collection_id": collection.id,
            "collection_name": collection.name,
            "documents": documents,
            "knowledge_version": self._version(),
        }

    def search(self, collection_id: str, query: str, top_k: int) -> dict[str, Any]:
        collection = self._resolve_collection(collection_id)
        retriever = getattr(self._deps, "retriever", None)
        if retriever is None:
            raise ToolFailure("retrieval_unavailable", "Search is unavailable right now.")
        try:
            hits: list[Hit] = list(retriever.search(collection.id, query, top_k))
        except RetrievalUnavailable:
            raise ToolFailure("retrieval_unavailable", "Search is unavailable right now.") from None
        except Exception:
            logger.exception("mcp search failed")
            raise ToolFailure("retrieval_unavailable", "Search is unavailable right now.") from None
        results = [
            {
                "rank": rank,
                "chunk_id": hit.chunk.id,
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "section": hit.chunk.section,
                "page": hit.chunk.page,
                "text": hit.chunk.text,
                "score": round(float(hit.score), 6),
            }
            for rank, hit in enumerate(hits[:top_k], start=1)
        ]
        if results:
            status = "ok"
        elif not self._has_documents(collection.id):
            status = "no_documents"
        else:
            status = "no_results"
        return {
            "collection_id": collection.id,
            "query": query,
            "status": status,
            "results": results,
            "knowledge_version": self._version(),
        }

    def ask(self, collection_id: str, question: str, top_k: int) -> dict[str, Any]:
        collection = self._resolve_collection(collection_id)
        started = time.perf_counter()
        version = self._version()
        hits: list[Hit] = []
        retriever = getattr(self._deps, "retriever", None)
        try:
            if retriever is None:
                raise RetrievalUnavailable("no retriever")
            hits = list(retriever.search(collection.id, question, top_k))
        except Exception as exc:
            if not isinstance(exc, RetrievalUnavailable):
                logger.exception("mcp ask: search failed")
            answer = compose_unavailable(question, collection_id=collection.id, knowledge_version=version)
        else:
            answer = compose(
                question,
                hits,
                self._extractive,
                self._deps.settings,
                collection_id=collection.id,
                knowledge_version=version,
                has_documents=self._has_documents(collection.id),
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        by_chunk = {hit.chunk.id: hit for hit in hits}
        passages = []
        for citation in answer.citations:
            hit = by_chunk.get(citation.chunk_id)
            passages.append(
                {
                    "marker": citation.marker,
                    "title": citation.title,
                    "section": citation.section,
                    "page": citation.page,
                    "text": hit.chunk.text if hit is not None else citation.snippet,
                }
            )

        try:
            self._meta.save_answer(answer.model_dump_json(), answer.id, collection.id)
        except Exception:
            logger.warning("could not store the MCP answer", exc_info=True)
        self._audit(
            "mcp_ask",
            collection_id=collection.id,
            answer_id=answer.id,
            question_chars=len(question),
            status=_status_value(answer.status),
            provider=self._extractive.name,
            elapsed_ms=elapsed_ms,
            passages=answer.passages_considered,
            citations=len(answer.citations),
            knowledge_version=answer.knowledge_version,
        )

        brief = answer.model_dump(mode="json")
        brief["passages"] = passages
        brief["guidance"] = GUIDANCE
        return brief


# ---------------------------------------------------------------- MCP wiring


def _call_result(payload: dict[str, Any], is_error: bool) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=is_error,
    )


def build_mcp_server(deps: Deps) -> Server:
    """The low-level MCP server with the four tools. Arguments are validated here, not by the SDK,
    so every refusal has the same `{"error": {...}}` shape."""
    tools = KnowledgeBaseTools(deps)

    async def on_list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=TOOLS)

    async def on_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
        payload, is_error = await anyio.to_thread.run_sync(tools.call, params.name, params.arguments)
        return _call_result(payload, is_error)

    def tool_input_schema(name: str) -> dict[str, Any] | None:
        tool = _TOOLS_BY_NAME.get(name)
        return None if tool is None else tool.input_schema

    return Server(
        SERVER_NAME,
        version=__version__,
        title="rag-kb knowledge base",
        instructions=INSTRUCTIONS,
        get_tool_input_schema=tool_input_schema,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(error_payload(code, message), status_code=status_code)


class McpEndpoint:
    """The ASGI app behind /mcp: the loopback guard, then the SDK's streamable HTTP transport.

    The session manager is created per app lifespan (`running()`), so the same app can start, stop
    and start again (as test clients do). Stateless with JSON responses: every POST stands alone,
    which suits tool-only servers, tunnels and several workers.
    """

    def __init__(self, server: Server, *, public: bool):
        self.server = server
        self.public = public
        self._manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        manager = StreamableHTTPSessionManager(
            app=self.server,
            json_response=True,
            stateless=True,
            # The Host and Origin checks are done by this class with a clearer error; the SDK still
            # validates the Content-Type of every POST.
            security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        async with manager.run():
            self._manager = manager
            try:
                yield
            finally:
                self._manager = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        headers = Headers(scope=scope)
        if not self.public:
            if not host_is_loopback(headers.get("host")):
                response = _error_response(
                    403,
                    "forbidden",
                    "The MCP endpoint only answers on this machine (localhost). To reach it from elsewhere, "
                    "set RAGKB_MCP_PUBLIC=true and put it behind a tunnel or your own sign-in.",
                )
                await response(scope, receive, send)
                return
            if not origin_is_loopback(headers.get("origin")):
                response = _error_response(
                    403, "forbidden", "Requests from web pages on other sites are not accepted."
                )
                await response(scope, receive, send)
                return
        manager = self._manager
        if manager is None:
            response = _error_response(503, "not_ready", "The MCP endpoint is not running yet. Try again shortly.")
            await response(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)


def mount_mcp(app: FastAPI, deps: Deps) -> None:
    """Mount the MCP server at /mcp (and /mcp/) on `app`.

    Call it while building the app, before it starts. The route goes first in the routing table, so a
    UI mounted at "/" cannot shadow it. The MCP session manager starts and stops with the app: through
    `app.state.lifespan_hooks` when the app factory provides that list, otherwise by wrapping the
    app's lifespan.
    """
    settings = deps.settings
    if not settings.MCP_ENABLED:
        logger.info("MCP is disabled (RAGKB_MCP_ENABLED=false); /mcp is not mounted")
        return
    endpoint = McpEndpoint(build_mcp_server(deps), public=settings.MCP_PUBLIC)
    asgi: ASGIApp = endpoint
    app.router.routes.insert(0, Route(MCP_PATH + "/", endpoint=asgi, include_in_schema=False))
    app.router.routes.insert(0, Route(MCP_PATH, endpoint=asgi, include_in_schema=False))

    hooks = getattr(app.state, "lifespan_hooks", None)
    if isinstance(hooks, list):
        # The app factory enters these async context managers inside its own lifespan.
        hooks.append(endpoint.running)
    else:
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_with_mcp(running_app: Any) -> AsyncIterator[Any]:
            async with endpoint.running(), original_lifespan(running_app) as state:
                yield state

        app.router.lifespan_context = lifespan_with_mcp
    app.state.mcp_endpoint = endpoint
    logger.info("MCP mounted at %s (%s)", MCP_PATH, "public" if settings.MCP_PUBLIC else "loopback only")
