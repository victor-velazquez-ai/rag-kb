"""The single-page UI: index.html at / and the assets (or index.html for deep links) under /ui/."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ragkb.server.errors import ApiError

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

# The page loads only its own files: no inline script, no external hosts.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}

router = APIRouter(include_in_schema=False)


def _serve(path: Path) -> FileResponse:
    return FileResponse(path, media_type=_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        headers=_HEADERS)


def _index() -> FileResponse:
    index = UI_DIR / "index.html"
    if not index.is_file():
        raise ApiError(404, "not_found", "The web interface is not installed.")
    return _serve(index)


@router.get("/")
def index() -> FileResponse:
    return _index()


@router.get("/ui")
@router.get("/ui/")
def ui_root() -> FileResponse:
    return _index()


@router.get("/ui/{asset:path}")
def ui_asset(asset: str) -> FileResponse:
    root = UI_DIR.resolve()
    candidate = (root / asset).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return _serve(candidate)
    if Path(asset).suffix:
        # A missing asset (a .js or .css file) is a real 404, not the page.
        raise ApiError(404, "not_found", "Not found.")
    return _index()
