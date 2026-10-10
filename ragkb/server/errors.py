"""The JSON error envelope.

Every error leaves the server as ``{"error": {"code": "...", "message": "..."}}``. Messages are plain
sentences; they never carry file paths or stack traces (those go to the server log only).
"""
from __future__ import annotations

import http
import logging
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("ragkb.server")

_CODE_BY_STATUS: dict[int, str] = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "file_too_large",
    415: "unsupported_file_type",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal",
    503: "unavailable",
}

_MESSAGE_BY_STATUS: dict[int, str] = {
    400: "The request is not valid.",
    401: "Sign in first.",
    403: "This request is not allowed.",
    404: "Not found.",
    405: "This method is not allowed for this address.",
    409: "This conflicts with something that already exists.",
    413: "The upload is too large.",
    415: "This file type is not supported.",
    422: "The request is not valid.",
    429: "Too many requests. Try again in a moment.",
    500: "Something went wrong on the server. The details are in the server log.",
    503: "This service is not available right now.",
}

INTERNAL_MESSAGE = _MESSAGE_BY_STATUS[500]


class ApiError(StarletteHTTPException):
    """An HTTP error with a stable machine code and a plain message."""

    def __init__(self, status_code: int, code: str, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def error_response(status_code: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message), headers=headers)


# Windows drive paths, UNC paths and POSIX paths with at least two segments.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|]*|(?<![\w.])/(?:[\w.\-]+/)+[\w.\-]*")
_TRACE_RE = re.compile(r"Traceback \(most recent call last\)")


def strip_paths(text: str) -> str:
    return _PATH_RE.sub("(path hidden)", text)


def plain_error(exc: BaseException, fallback: str, *, limit: int = 300) -> str:
    """A message safe to show a user: our own errors keep their text, anything else gets the fallback.

    Paths are removed and the text is capped, so a library error can never leak a location or a trace.
    """
    message: str | None = None
    own = type(exc).__module__.startswith("ragkb.")
    candidate = getattr(exc, "message", None)
    if isinstance(candidate, str) and candidate.strip() and own:
        message = candidate
    elif own:
        message = str(exc)
    if not message or _TRACE_RE.search(message):
        return fallback
    message = " ".join(strip_paths(message).split())
    if len(message) > limit:
        message = message[: limit - 3].rstrip() + "..."
    return message or fallback


def _validation_message(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return _MESSAGE_BY_STATUS[422]
    first = errors[0]
    kind = str(first.get("type", ""))
    if kind in {"json_invalid", "json_type"}:
        return "The request body is not valid JSON."
    loc = [str(part) for part in first.get("loc", ()) if part not in ("body", "query", "path", "form", "header")]
    field = ".".join(loc)
    if kind == "missing":
        return f"{field} is required." if field else "A required value is missing."
    msg = str(first.get("msg", "is not valid")).rstrip(".")
    msg = strip_paths(msg)
    return f"{field}: {msg}." if field else f"{msg}."


async def _http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    status = exc.status_code
    code = getattr(exc, "code", None) or _CODE_BY_STATUS.get(status, "error")
    message = getattr(exc, "message", None)
    if not message:
        detail = exc.detail
        try:
            phrase = http.HTTPStatus(status).phrase
        except ValueError:
            phrase = ""
        if isinstance(detail, str) and detail and detail != phrase:
            message = strip_paths(detail)
        else:
            message = _MESSAGE_BY_STATUS.get(status, "The request could not be completed.")
    return error_response(status, code, message, headers=getattr(exc, "headers", None))


async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "invalid_request", _validation_message(list(exc.errors())))


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(
        "unhandled error",
        exc_info=(type(exc), exc, exc.__traceback__),
        extra={"method": request.method, "path": request.url.path},
    )
    return error_response(500, "internal", INTERNAL_MESSAGE)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)
