"""Collections: list, create, delete with all documents, and stats.

Counts follow one rule everywhere: ``documents`` is the number of ready (searchable) documents;
``documents_total`` and ``documents_by_status`` carry the rest.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, field_validator

from ragkb.models import Collection
from ragkb.server.deps import DELETE_PENDING_MESSAGE, CollectionOut, DepsDep
from ragkb.server.text import clean_text

router = APIRouter(tags=["collections"])


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        value = clean_text(value)
        if not value:
            raise ValueError("name must have visible characters")
        return value

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        return clean_text(value)


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(deps: DepsDep) -> list[CollectionOut]:
    return deps.list_collections()


@router.post("/collections", response_model=Collection, status_code=201)
def create_collection(body: CollectionCreate, deps: DepsDep) -> Collection:
    return deps.create_collection(body.name, body.description)


@router.delete("/collections/{cid}")
def delete_collection(cid: str, deps: DepsDep, response: Response) -> dict[str, Any]:
    """200 when the collection is gone; 202 with ``pending`` when the delete continues in the background."""
    removed, done = deps.delete_collection(cid)
    if not done:
        response.status_code = 202
        return {"deleted": cid, "documents_deleted": removed, "pending": True, "message": DELETE_PENDING_MESSAGE}
    return {"deleted": cid, "documents_deleted": removed, "knowledge_version": deps.knowledge_version()}


@router.get("/collections/{cid}/stats")
def collection_stats(cid: str, deps: DepsDep) -> dict[str, Any]:
    return deps.collection_stats(cid)
