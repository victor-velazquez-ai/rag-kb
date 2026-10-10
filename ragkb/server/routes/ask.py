"""POST /api/collections/{cid}/ask and GET /api/answers/{aid}.

Degraded outcomes (not covered, no documents, provider not signed in, provider quota, search down)
come back as an ``Answer`` with a status, not as HTTP errors, so every client can render them. The
only HTTP-level refusal is this server's own per-collection rate limit (429 ``rate_limited``).
"""
from __future__ import annotations

import math
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from ragkb.models import Answer
from ragkb.server.deps import DepsDep
from ragkb.server.errors import ApiError

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    provider: Literal["auto", "claude_code", "codex", "extractive"] | None = None
    # A model alias or id. It must start with a letter or digit so it can never read as a CLI flag.
    model: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/\[\]-]*$")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


@router.post("/collections/{cid}/ask", response_model=Answer)
def ask(cid: str, body: AskRequest, deps: DepsDep) -> Answer:
    deps.get_collection(cid)
    wait = deps.ask_limiter.check(cid)
    if wait is not None:
        seconds = max(1, math.ceil(wait))
        raise ApiError(
            429,
            "rate_limited",
            f"Too many questions for this collection in the last minute (limit {deps.ask_limiter.limit} per "
            f"minute on this server process). Try again in {seconds} seconds.",
            headers={"Retry-After": str(seconds)},
        )
    return deps.ask(cid, body.question, top_k=body.top_k, provider=body.provider, model=body.model or None)


@router.get("/answers/{aid}", response_model=Answer)
def get_answer(aid: str, deps: DepsDep) -> Answer:
    return deps.get_answer(aid)
