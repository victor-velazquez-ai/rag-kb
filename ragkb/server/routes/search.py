"""POST /api/collections/{cid}/search: ranked passages, no model call."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from ragkb.models import Hit
from ragkb.server.deps import DepsDep

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


@router.post("/collections/{cid}/search", response_model=list[Hit])
def search(cid: str, body: SearchRequest, deps: DepsDep) -> list[Hit]:
    return deps.search(cid, body.query, body.top_k or deps.settings.TOP_K)
