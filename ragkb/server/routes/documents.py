"""Documents and ingestion jobs: upload, list, passages, original file, re-index, retry, delete, progress,
cancel and the per-collection job summary."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ragkb.models import Chunk, Document
from ragkb.server.deps import DELETE_PENDING_MESSAGE, DepsDep, JobOut
from ragkb.server.errors import ApiError
from ragkb.server.text import clean_filename, clean_text
from ragkb.server.uploads import Incoming, Rejection, accept_uploads, is_archive

router = APIRouter(tags=["documents"])


class UploadResponse(BaseModel):
    """Every file of the request is in exactly one list: ``jobs`` (accepted) or ``rejected`` (refused up
    front, with a code; no document row). ``skipped`` counts zip entries left out on purpose (folders,
    hidden files)."""

    jobs: list[JobOut]
    rejected: list[Rejection] = []
    skipped: int = 0


@router.post("/collections/{cid}/documents", response_model=UploadResponse, status_code=202)
async def upload_documents(
    cid: str,
    files: Annotated[list[UploadFile], File(description="One or many files. A .zip archive is expanded.")],
    deps: DepsDep,
    title: Annotated[str | None, Form(max_length=300)] = None,
    relative_paths: Annotated[
        list[str] | None,
        Form(description="For a folder upload: each file's path inside the folder, in the same order as files."),
    ] = None,
) -> UploadResponse:
    deps.require("meta")
    deps.require("pipeline")
    await run_in_threadpool(deps.get_collection, cid)
    clean_title: str | None = None
    if title is not None and title.strip():
        clean_title = clean_text(title)
        if not clean_title:
            raise ApiError(400, "invalid_request", "The title has no visible characters.")
    if relative_paths is not None and len(relative_paths) != len(files):
        raise ApiError(400, "invalid_request",
                       "relative_paths needs one entry per file, in the same order as files.")
    if clean_title and (len(files) > 1 or is_archive(clean_filename(files[0].filename))):
        raise ApiError(400, "invalid_request", "A title can only be set when uploading a single file.")
    items = [
        Incoming(upload.filename or "", upload.file, (relative_paths[i] or None) if relative_paths else None)
        for i, upload in enumerate(files)
    ]
    try:
        outcome = await run_in_threadpool(accept_uploads, deps, cid, items, clean_title)
    finally:
        for upload in files:
            await upload.close()
    if not outcome.jobs:
        if not outcome.rejected:
            raise ApiError(400, "invalid_request", "The upload holds no files to index.")
        first = outcome.rejected[0]
        message = first.message
        if len(outcome.rejected) > 1:
            message = f"None of the {len(outcome.rejected)} files could be added. {first.message}"
        raise ApiError(outcome.first_status, first.code, message)
    return UploadResponse(jobs=outcome.jobs, rejected=outcome.rejected, skipped=outcome.skipped)


@router.get("/collections/{cid}/documents", response_model=list[Document])
def list_documents(cid: str, deps: DepsDep) -> list[Document]:
    deps.get_collection(cid)
    docs = deps.meta.list_documents(cid)
    return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)


@router.get("/collections/{cid}/jobs", response_model=list[JobOut])
def list_jobs(cid: str, deps: DepsDep, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[JobOut]:
    return deps.list_jobs(cid, limit=limit)


@router.get("/collections/{cid}/jobs/summary")
def jobs_summary(
    cid: str,
    deps: DepsDep,
    since: Annotated[datetime | None, Query(description="ISO 8601 time; the default is 24 hours ago.")] = None,
) -> dict[str, Any]:
    """How the collection's recent jobs ended: queued, processing, ready, failed, cancelled, replaced,
    duplicates and total. Each job counts once."""
    return deps.jobs_summary(cid, since)


@router.get("/jobs/{jid}", response_model=JobOut)
def get_job(jid: str, deps: DepsDep) -> JobOut:
    return deps.get_job(jid)


@router.post("/jobs/{jid}/cancel", response_model=JobOut)
def cancel_job(jid: str, deps: DepsDep) -> JobOut:
    """A waiting job ends as cancelled at once; a running one stops at its next step and keeps any
    previous version. A finished job comes back unchanged."""
    return deps.cancel_job(jid)


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


@router.post("/documents/{did}/retry", response_model=JobOut, status_code=202)
def retry_document(did: str, deps: DepsDep) -> JobOut:
    """Index a failed document again from its kept original; its error is cleared while it waits."""
    return deps.submit_reindex(did, retry=True)


@router.delete("/documents/{did}")
def delete_document(did: str, deps: DepsDep, response: Response) -> dict[str, Any]:
    """200 when the document is gone; 202 with ``pending`` when the delete continues in the background."""
    if not deps.delete_document(did):
        response.status_code = 202
        return {"deleted": did, "pending": True, "message": DELETE_PENDING_MESSAGE}
    return {"deleted": did, "knowledge_version": deps.knowledge_version()}
