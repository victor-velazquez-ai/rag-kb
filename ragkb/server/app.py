"""FastAPI app factory: routes, the JSON error envelope, structured logs, the UI and the MCP mount."""
from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragkb import __version__
from ragkb.config import Settings, get_settings
from ragkb.server.deps import Deps, build_deps
from ragkb.server.errors import error_response, install_error_handlers
from ragkb.server.loopback import host_is_loopback
from ragkb.server.routes import ask, collections, documents, feedback, health, providers, search, ui

try:
    from ragkb.mcp.server import mount_mcp
except Exception as _exc:  # noqa: BLE001 - a missing or broken part must not stop the app
    mount_mcp = None  # type: ignore[assignment]
    _MCP_IMPORT_PROBLEM: str | None = f"{type(_exc).__name__}: {_exc}"
else:
    _MCP_IMPORT_PROBLEM = None

log = logging.getLogger("ragkb.server")

_RESERVED_LOG_KEYS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_QUIET_PATHS = {"/api/health"}
_HEALTH_PATHS = {"/api/health", "/api/health/"}
API_LOCAL_ONLY_MESSAGE = (
    "The API only answers on this machine (localhost). To reach it from elsewhere, set "
    "RAGKB_API_PUBLIC=true and put it behind your own sign-in."
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message and any extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach JSON handlers to the ``ragkb`` logger. Safe to call more than once."""
    logger = logging.getLogger("ragkb")
    for handler in list(logger.handlers):
        if getattr(handler, "_ragkb", False):
            logger.removeHandler(handler)
            handler.close()
    stream = logging.StreamHandler(sys.stderr)
    handlers: list[logging.Handler] = [stream]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        handler._ragkb = True
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _same_origin(origin: str, host: str | None) -> bool:
    if not host or origin == "null":
        return False
    return urlsplit(origin).netloc.lower() == host.lower()


class RequestMiddleware:
    """Logs every request with its status and time, keeps the API on this machine, and refuses
    cross-site writes to the API.

    Unless ``api_public`` is set (``RAGKB_API_PUBLIC=true``), a request to ``/api`` whose Host is not
    loopback gets 403: a tunnel opened for an assistant to reach ``/mcp`` must not expose the API,
    which has no sign-in. ``GET /api/health`` stays open for monitoring.

    A browser always sends ``Origin`` on a cross-site POST or DELETE, so a page on another site cannot
    upload, ask or delete through a user's local server. Same-origin calls and non-browser clients
    (no ``Origin`` header) pass.
    """

    def __init__(self, app: ASGIApp, api_public: bool = False) -> None:
        self.app = app
        self.api_public = api_public

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        is_api = path == "/api" or path.startswith("/api/")
        health = method == "GET" and path in _HEALTH_PATHS
        if is_api and not self.api_public and not health and not host_is_loopback(_header(scope, b"host")):
            log.warning("non-local API request refused", extra={"method": method, "path": path})
            response = error_response(403, "forbidden", API_LOCAL_ONLY_MESSAGE)
            await response(scope, receive, send)
            return
        if method in _UNSAFE_METHODS and path.startswith("/api/"):
            origin = _header(scope, b"origin")
            if origin is not None and not _same_origin(origin, _header(scope, b"host")):
                log.warning("cross-site request refused", extra={"method": method, "path": path})
                response = error_response(403, "forbidden", "Cross-site requests are not allowed.")
                await response(scope, receive, send)
                return
        started = time.perf_counter()
        state = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            level = logging.DEBUG if path in _QUIET_PATHS or not path.startswith(("/api", "/mcp")) else logging.INFO
            log.log(level, "request", extra={"method": method, "path": path, "status": state["status"],
                                             "elapsed_ms": elapsed_ms})


def _lifespan_factory(deps: Deps) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("rag-kb started", extra={"version": __version__, "status": deps.health()["status"]})
        async with AsyncExitStack() as stack:
            # Parts that mount sub-apps needing startup work (for example the MCP session manager) can
            # append async context manager factories to app.state.lifespan_hooks.
            for hook in list(getattr(app.state, "lifespan_hooks", [])):
                await stack.enter_async_context(hook())
            try:
                yield
            finally:
                deps.close()
                log.info("rag-kb stopped")

    return lifespan


def create_app(settings: Settings | None = None, *, deps: Deps | None = None) -> FastAPI:
    """Build the app. Pass ``deps`` to reuse a container (tests, the CLI); otherwise one is built here."""
    settings = settings or (deps.settings if deps is not None else get_settings())
    configure_logging(settings.LOG_LEVEL, settings.logs_dir / "ragkb.log")
    deps = deps or build_deps(settings)

    app = FastAPI(
        title="rag-kb",
        version=__version__,
        description="Enterprise knowledge base with retrieval-augmented, cited answers.",
        lifespan=_lifespan_factory(deps),
    )
    app.state.deps = deps
    app.state.lifespan_hooks = []
    install_error_handlers(app)
    app.add_middleware(RequestMiddleware, api_public=settings.API_PUBLIC)

    for module in (health, providers, collections, documents, search, ask, feedback):
        app.include_router(module.router, prefix="/api")

    if settings.MCP_ENABLED:
        if mount_mcp is None:
            log.warning("MCP server not available", extra={"reason": _MCP_IMPORT_PROBLEM})
            deps.problems["mcp"] = "not installed"
        else:
            try:
                mount_mcp(app, deps)
                deps.mcp_mounted = True
            except Exception:
                log.warning("could not mount the MCP server", exc_info=True)
                deps.problems["mcp"] = "could not start"

    if settings.UI_ENABLED:
        app.include_router(ui.router)

    return app
