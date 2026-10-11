"""Feedback on answers: record a thumbs up or down with an optional note, and read the votes back."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ragkb.models import Feedback
from ragkb.server.deps import DepsDep

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    note: str = Field(default="", max_length=2000)


@router.post("/answers/{aid}/feedback", response_model=Feedback, status_code=201)
def add_feedback(aid: str, body: FeedbackRequest, deps: DepsDep) -> Feedback:
    return deps.add_feedback(aid, body.rating, body.note)


@router.get("/answers/{aid}/feedback", response_model=list[Feedback])
def answer_feedback(aid: str, deps: DepsDep) -> list[Feedback]:
    """The votes on one answer, newest first."""
    return deps.answer_feedback(aid)


@router.get("/collections/{cid}/feedback", response_model=list[Feedback])
def collection_feedback(cid: str, deps: DepsDep,
                        limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[Feedback]:
    """A collection's votes, newest first, each with the question it was about."""
    return deps.list_feedback(cid, limit=limit)
