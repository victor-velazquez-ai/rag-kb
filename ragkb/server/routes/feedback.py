"""POST /api/answers/{aid}/feedback: thumbs up or down with an optional note."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ragkb.models import Feedback
from ragkb.server.deps import DepsDep, utcnow

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    note: str = Field(default="", max_length=2000)


@router.post("/answers/{aid}/feedback", response_model=Feedback, status_code=201)
def add_feedback(aid: str, body: FeedbackRequest, deps: DepsDep) -> Feedback:
    meta = deps.require("meta")
    answer = deps.get_answer(aid)
    item = Feedback(answer_id=answer.id, rating=body.rating, note=body.note.strip(), created_at=utcnow())
    meta.add_feedback(item)
    deps.audit("feedback", collection_id=answer.collection_id, answer_id=answer.id, rating=body.rating,
               has_note=bool(item.note))
    return item
