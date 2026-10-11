"""From an upload request to jobs: the per-file rules, folder paths and zip archives.

Every file in a request ends in exactly one place in the response: a job (the file was accepted; if
it then fails to index, its document row says failed, with the reason) or a rejection with a code
(it broke a rule up front and left no row: an unsupported type, too large, empty, or password
protected). Zip archives are expanded here, one entry at a time, so memory holds one file at a time;
their folders and hidden files are skipped.
"""
from __future__ import annotations

import logging
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import BinaryIO

from pydantic import BaseModel

from ragkb.server.deps import Deps, JobOut, encrypted_reason, supported_media_type, supported_types
from ragkb.server.text import (
    clean_filename,
    clean_relative_path,
    clean_text,
    is_hidden_path,
    short_name,
    suffix_of,
)

log = logging.getLogger("ragkb.server")

MB = 1024 * 1024
ARCHIVE_SUFFIXES = frozenset({".zip"})
# An archive may expand to at most this many times the per-file limit, in total.
ZIP_EXPANSION_FACTOR = 10
MAX_ZIP_ENTRIES = 10_000
_ENCRYPTED = 0x1


class Rejection(BaseModel):
    """A file that was refused up front. No document row exists for it."""

    filename: str
    source_path: str = ""
    code: str
    message: str


@dataclass
class Incoming:
    """One file part of the request, still in the request's temporary file."""

    filename: str
    file: BinaryIO
    relative_path: str | None = None


@dataclass
class UploadOutcome:
    jobs: list[JobOut] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    # zip entries left out on purpose: folders, hidden files and archive metadata
    skipped: int = 0
    # the HTTP status of the first rejection, for a request where nothing was accepted
    first_status: int = 400

    def reject(self, status: int, code: str, filename: str, message: str, source_path: str = "") -> None:
        if not self.rejected:
            self.first_status = status
        self.rejected.append(Rejection(filename=filename, source_path=source_path, code=code, message=message))


def is_archive(name: str) -> bool:
    return suffix_of(name) in ARCHIVE_SUFFIXES


def unsupported_message(name: str, kinds: list[str]) -> str:
    suffix = suffix_of(name)
    kind = f"Files of type {suffix} are" if suffix else "Files without an extension are"
    listed = ", ".join([*kinds, *sorted(ARCHIVE_SUFFIXES)]) or "none"
    return f"{kind} not supported ({short_name(name)}). Supported types: {listed}."


def too_large_message(name: str, limit_mb: int) -> str:
    return f"The file is larger than the {limit_mb} MB limit per file ({short_name(name)})."


def empty_message(name: str) -> str:
    return f"The file is empty ({short_name(name)})."


def outside_message(name: str) -> str:
    return f"The file's folder path points outside the uploaded folder ({short_name(name)})."


def accept_uploads(deps: Deps, collection_id: str, items: list[Incoming], title: str | None) -> UploadOutcome:
    """Check each file, expand archives and queue a job per accepted file. Blocking: run it in a thread."""
    cap = deps.settings.MAX_UPLOAD_MB * MB
    outcome = UploadOutcome()
    for item in items:
        raw_name = item.filename or ""
        name = clean_filename(raw_name)
        if not name:
            outcome.reject(400, "invalid_request", clean_text(raw_name), "Every uploaded file needs a name.")
            continue
        try:
            source_path = clean_relative_path(item.relative_path) if item.relative_path else ""
        except ValueError:
            outcome.reject(400, "invalid_request", name, outside_message(item.relative_path or name))
            continue
        if is_archive(name):
            _expand_archive(deps, collection_id, name, source_path, item.file, cap, outcome)
            continue
        data = _check_file(deps, name, source_path, item.file, cap, outcome)
        if data is not None:
            outcome.jobs.append(deps.submit_upload(collection_id, name, data, title=title,
                                                   source_path=source_path or None))
    return outcome


def _check_file(deps: Deps, name: str, source_path: str, stream: BinaryIO, cap: int,
                outcome: UploadOutcome) -> bytes | None:
    """The per-file rules (type, size, not empty, not password protected). Returns the bytes, or None
    after recording a rejection."""
    if supported_media_type(name) is None:
        outcome.reject(415, "unsupported_file_type", name, unsupported_message(name, supported_types()), source_path)
        return None
    data = stream.read(cap + 1)
    if len(data) > cap:
        outcome.reject(413, "file_too_large", name, too_large_message(name, deps.settings.MAX_UPLOAD_MB), source_path)
        return None
    if not data:
        outcome.reject(400, "empty_document", name, empty_message(name), source_path)
        return None
    if not _not_encrypted(name, source_path, data, outcome):
        return None
    return data


def _not_encrypted(name: str, source_path: str, data: bytes, outcome: UploadOutcome) -> bool:
    """False, after recording an ``encrypted_file`` rejection, for a password-protected PDF, Word or Excel
    file: it is refused before it is queued, so it leaves no job and no document row."""
    reason = encrypted_reason(name, data)
    if reason is None:
        return True
    outcome.reject(400, "encrypted_file", name, f"{reason.rstrip('.')} ({short_name(name)}).", source_path)
    return False


def _expand_archive(deps: Deps, collection_id: str, name: str, source_path: str, stream: BinaryIO, cap: int,
                    outcome: UploadOutcome) -> None:
    """One job per supported entry, with ``source_path`` set to the entry's path inside the archive.

    The archive is refused as a whole when it cannot be opened, holds too many entries, or would expand
    to more than ``ZIP_EXPANSION_FACTOR`` times the per-file limit. Entries that climb out of the
    archive's folder, are encrypted, are archives themselves, or break the per-file rules are refused
    one by one. Folders, hidden files and macOS metadata are skipped.
    """
    total_cap = cap * ZIP_EXPANSION_FACTOR
    limit_mb = deps.settings.MAX_UPLOAD_MB * ZIP_EXPANSION_FACTOR
    archive_path = source_path or name
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(0)
    if size == 0:
        outcome.reject(400, "empty_document", name, empty_message(name), archive_path)
        return
    if size > total_cap:
        outcome.reject(413, "file_too_large", name,
                       f"The archive is larger than the {limit_mb} MB limit for one archive ({short_name(name)}).",
                       archive_path)
        return
    try:
        archive = zipfile.ZipFile(stream)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        outcome.reject(400, "invalid_request", name,
                       f"The archive could not be opened; it may be damaged or not a zip file ({short_name(name)}).",
                       archive_path)
        return
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            outcome.reject(400, "invalid_request", name,
                           f"The archive holds more than {MAX_ZIP_ENTRIES} entries; split it into smaller archives "
                           f"({short_name(name)}).", archive_path)
            return
        planned, refused, skipped = _plan_entries(deps, entries, cap)
        expanded = sum(info.file_size for info, _, _ in planned)
        if expanded > total_cap:
            outcome.reject(413, "file_too_large", name,
                           f"The archive expands to more than the {limit_mb} MB limit for one archive "
                           f"({short_name(name)}).", archive_path)
            return
        outcome.skipped += skipped
        for status, code, entry_name, message, entry_path in refused:
            outcome.reject(status, code, entry_name, message, entry_path)
        for info, entry_name, entry_path in planned:
            data = _read_entry(archive, info, entry_name, entry_path, cap, deps, outcome)
            if data is not None and _not_encrypted(entry_name, entry_path, data, outcome):
                outcome.jobs.append(deps.submit_upload(collection_id, entry_name, data, source_path=entry_path))


def _plan_entries(deps: Deps, entries: list[zipfile.ZipInfo], cap: int) -> tuple[
        list[tuple[zipfile.ZipInfo, str, str]], list[tuple[int, str, str, str, str]], int]:
    """Sort the entries into (to read, refused, number skipped) from the central directory alone."""
    planned: list[tuple[zipfile.ZipInfo, str, str]] = []
    refused: list[tuple[int, str, str, str, str]] = []
    skipped = 0
    for info in entries:
        if info.is_dir():
            skipped += 1
            continue
        try:
            path = clean_relative_path(info.filename)
        except ValueError:
            shown = clean_text(info.filename.replace("\\", "/"))
            refused.append((400, "invalid_request", clean_filename(shown) or shown,
                            f"This entry points outside the archive's folder ({short_name(shown)}).", ""))
            continue
        if not path or is_hidden_path(path):
            skipped += 1
            continue
        entry_name = clean_filename(path)
        if not entry_name:
            skipped += 1
            continue
        if info.flag_bits & _ENCRYPTED:
            refused.append((400, "encrypted_file", entry_name,
                            f"This entry is password protected and cannot be read ({short_name(entry_name)}).", path))
        elif is_archive(entry_name):
            refused.append((415, "unsupported_file_type", entry_name,
                            f"Archives inside an archive are not expanded ({short_name(entry_name)}).", path))
        elif supported_media_type(entry_name) is None:
            refused.append((415, "unsupported_file_type", entry_name,
                            unsupported_message(entry_name, supported_types()), path))
        elif info.file_size > cap:
            refused.append((413, "file_too_large", entry_name,
                            too_large_message(entry_name, deps.settings.MAX_UPLOAD_MB), path))
        elif info.file_size == 0:
            refused.append((400, "empty_document", entry_name, empty_message(entry_name), path))
        else:
            planned.append((info, entry_name, path))
    return planned, refused, skipped


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo, entry_name: str, entry_path: str, cap: int,
                deps: Deps, outcome: UploadOutcome) -> bytes | None:
    """An entry's bytes, never more than the per-file limit (the zip reader stops at the declared size)."""
    try:
        with archive.open(info) as handle:
            data = handle.read(cap + 1)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError, EOFError, zlib.error):
        log.info("an archive entry could not be read", exc_info=True)
        outcome.reject(400, "invalid_request", entry_name,
                       f"This entry could not be read from the archive; it may be damaged ({short_name(entry_name)}).",
                       entry_path)
        return None
    if len(data) > cap:
        outcome.reject(413, "file_too_large", entry_name, too_large_message(entry_name, deps.settings.MAX_UPLOAD_MB),
                       entry_path)
        return None
    if not data:
        outcome.reject(400, "empty_document", entry_name, empty_message(entry_name), entry_path)
        return None
    return data
