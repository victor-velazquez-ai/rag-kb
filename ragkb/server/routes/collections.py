"""Collections: list, create, delete with all documents, and stats."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from ragkb.models import Collection
from ragkb.server.deps import CollectionOut, DepsDep

router = APIRouter(tags=["collections"])


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


@router.get("/collections", response_model=list[CollectionOut])
def list_collections(deps: DepsDep) -> list[CollectionOut]:
    return deps.list_collections()


@router.post("/collections", response_model=Collection, status_code=201)
def create_collection(body: CollectionCreate, deps: DepsDep) -> Collection:
    return deps.create_collection(body.name, body.description)


@router.delete("/collections/{cid}")
def delete_collection(cid: str, deps: DepsDep) -> dict[str, Any]:
    removed = deps.delete_collection(cid)
    return {"deleted": cid, "documents_deleted": removed, "knowledge_version": deps.knowledge_version()}


@router.get("/collections/{cid}/stats")
def collection_stats(cid: str, deps: DepsDep) -> dict[str, Any]:
    return deps.collection_stats(cid)
