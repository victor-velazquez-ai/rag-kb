"""Plain-text hygiene for names that people type or that arrive with uploads.

Collection names and descriptions, document titles, file names and folder paths lose control
characters (C0, C1 and DEL) and the Unicode bidirectional controls, which can hide or reorder text
(``abc`` + RIGHT-TO-LEFT OVERRIDE + ``def`` renders as ``abcfed``). Line breaks and tabs become spaces
and runs of whitespace collapse to one space.
"""
from __future__ import annotations

import re

# The Unicode Bidi_Control set: ALM, LRM, RLM, LRE to RLO, LRI to PDI.
_BIDI_CONTROLS: tuple[int, ...] = (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A))
_CONTROLS: tuple[int, ...] = (*range(0x20), *range(0x7F, 0xA0))
# Whitespace controls (tab, line feed, NEL and the like) become a space; every other one is removed.
_TABLE: dict[int, str | None] = {
    code: (" " if chr(code).isspace() else None) for code in (*_CONTROLS, *_BIDI_CONTROLS)
}
_DRIVE = re.compile(r"[A-Za-z]:")

ELLIPSIS = "..."
MESSAGE_NAME_LIMIT = 60
MAX_FILENAME_CHARS = 255
MAX_RELATIVE_PATH_CHARS = 1000


def clean_text(value: str | None) -> str:
    """Control and bidi characters removed, whitespace collapsed, ends trimmed."""
    if not value:
        return ""
    return " ".join(value.translate(_TABLE).split())


def short_name(name: str | None, limit: int = MESSAGE_NAME_LIMIT) -> str:
    """A name for a message: at most ``limit`` characters, keeping the end so the extension shows."""
    text = clean_text(name)
    if len(text) <= limit:
        return text
    tail = min(16, limit // 3)
    head = max(1, limit - len(ELLIPSIS) - tail)
    return text[:head].rstrip() + ELLIPSIS + text[-tail:].lstrip()


def clean_filename(raw: str | None) -> str:
    """The base name of an uploaded file (no folders), cleaned; empty when nothing usable is left."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = clean_text(name).strip(" .")
    if len(name) <= MAX_FILENAME_CHARS:
        return name
    suffix = suffix_of(name)[:16]  # a long name keeps its extension, so its type is still known
    return name[: MAX_FILENAME_CHARS - len(suffix)].rstrip(" .") + suffix


def clean_relative_path(raw: str | None) -> str:
    """A folder-relative path with forward slashes and cleaned parts, or an empty string.

    Raises ValueError when the path is absolute, names a drive, climbs out of its folder with a
    ``..`` part, or is unreasonably long.
    """
    text = (raw or "").replace("\\", "/")
    if text.startswith("/"):
        raise ValueError("absolute path")
    parts: list[str] = []
    for raw_part in text.split("/"):
        part = clean_text(raw_part)
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("parent folder")
        parts.append(part)
    if parts and _DRIVE.match(parts[0]):
        raise ValueError("drive")
    path = "/".join(parts)
    if len(path) > MAX_RELATIVE_PATH_CHARS:
        raise ValueError("too long")
    return path


def is_hidden_path(path: str) -> bool:
    """True when any part of a relative path is hidden (starts with a dot) or is macOS archive metadata."""
    parts = [p for p in path.split("/") if p]
    return any(p.startswith(".") for p in parts) or (bool(parts) and parts[0] == "__MACOSX")


def suffix_of(name: str) -> str:
    """The lower-case extension with its dot, or an empty string."""
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
