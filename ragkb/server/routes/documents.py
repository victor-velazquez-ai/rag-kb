"""Documents and ingestion jobs: upload, list, passages, original file, re-index, delete, progress."""
from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ragkb.models import Chunk, Document
from ragkb.server.deps import DepsDep, JobOut, supported_media_type, supported_types
from ragkb.server.errors import ApiError

router = APIRouter(tags=["documents"])

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class UploadResponse(BaseModel):
    jobs: list[JobOut]


def clean_filename(raw: str | None) -> str:
    """Keep only the base name, without folders or control characters."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL.sub("", name).strip().strip(".")
    return name[:255]


def _suffix(name: str) -> str:
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


@router.post("/collections/{cid}/documents", response_model=UploadResponse, status_code=202)
async def upload_documents(
    cid: str,
    files: Annotated[list[UploadFile], File(description="One or many files.")],
    deps: DepsDep,
    title: Annotated[str | None, Form(max_length=300)] = None,
) -> UploadResponse:
    deps.require("meta")
    deps.require("pipeline")
    await run_in_threadpool(deps.get_collection, cid)
    title = " ".join(title.split()) if title else None
    if title and len(files) > 1:
        raise ApiError(400, "invalid_request", "A title can only be set when uploading a single file.")
    cap = deps.settings.MAX_UPLOAD_MB * 1024 * 1024
    staged: list[tuple[str, bytes]] = []
    for upload in files:
        name = clean_filename(upload.filename)
        if not name:
            raise ApiError(400, "invalid_request", "Every uploaded file needs a name.")
        if supported_media_type(name) is None:
            kinds = ", ".join(supported_types()) or "none"
            raise ApiError(415, "unsupported_file_type",
                           f"{name}: files of type {_suffix(name) or 'unknown'} are not supported. "
                           f"Supported: {kinds}.")
        data = await upload.read(cap + 1)
        await upload.close()
        if len(data) > cap:
            raise ApiError(413, "file_too_large",
                           f"{name} is larger than the {deps.settings.MAX_UPLOAD_MB} MB limit per file.")
        if not data:
            raise ApiError(400, "empty_document", f"{name} is empty.")
        staged.append((name, data))
    jobs = [await run_in_threadpool(deps.submit_upload, cid, name, data, title) for name, data in staged]
    return UploadResponse(jobs=jobs)


@router.get("/collections/{cid}/documents", response_model=list[Document])
def list_documents(cid: str, deps: DepsDep) -> list[Document]:
    deps.get_collection(cid)
    docs = deps.meta.list_documents(cid)
    return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)


@router.get("/collections/{cid}/jobs", response_model=list[JobOut])
def list_jobs(cid: str, deps: DepsDep, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[JobOut]:
    return deps.list_jobs(cid, limit=limit)


@router.get("/jobs/{jid}", response_model=JobOut)
def get_job(jid: str, deps: DepsDep) -> JobOut:
    return deps.get_job(jid)


@router.get("/documents/{did}", response_model=Document)
def get_document(did: str, deps: DepsDep) -> Document:
    return deps.get_document(did)


@router.get("/documents/{did}/chunks", response_model=list[Chunk])
def document_chunks(did: str, deps: DepsDep) -> list[Chunk]:
    deps.get_document(did)
    store = deps.require("store")
    try:
        chunks = store.chunks_for(did)
    except Exception:  # noqa: BLE001 - a store failure is reported as unavailable, never a crash
        raise ApiError(503, "retrieval_unavailable", "Passages cannot be read right now.") from None
    return sorted(chunks, key=lambda c: c.ordinal)


@router.get("/documents/{did}/file")
def document_file(did: str, deps: DepsDep) -> FileResponse:
    doc = deps.get_document(did)
    path = deps.original_path(doc)
    if path is None:
        raise ApiError(404, "not_found", "The original file is not available.")
    # Always a download, never rendered in our origin, so an uploaded HTML file cannot run here.
    headers = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"}
    return FileResponse(path, media_type=doc.media_type or "application/octet-stream", filename=doc.filename,
                        headers=headers, content_disposition_type="attachment")


@router.post("/documents/{did}/reindex", response_model=JobOut, status_code=202)
def reindex_document(did: str, deps: DepsDep) -> JobOut:
    return deps.submit_reindex(did)


@router.delete("/documents/{did}")
def delete_document(did: str, deps: DepsDep) -> dict[str, Any]:
    deps.delete_document(did)
    return {"deleted": did, "knowledge_version": deps.knowledge_version()}
