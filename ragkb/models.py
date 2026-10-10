"""Shared data shapes. Every layer imports these; see docs/ARCHITECTURE.md section 3."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Collection(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime


class DocumentStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(BaseModel):
    id: str
    collection_id: str
    title: str
    filename: str
    media_type: str
    size_bytes: int
    content_hash: str
    pages: int | None = None
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.queued
    error: str | None = None
    uploaded_at: datetime
    indexed_at: datetime | None = None
    knowledge_version: str | None = None
    metadata: dict = Field(default_factory=dict)


class Block(BaseModel):
    """Parser output: one structural unit of a document."""

    text: str
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    kind: Literal["paragraph", "heading", "table", "list", "code"] = "paragraph"


class Chunk(BaseModel):
    id: str
    doc_id: str
    collection_id: str
    ordinal: int
    text: str
    embed_text: str
    title: str
    section: str = ""
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    knowledge_version: str = ""
    created_at: datetime


class Hit(BaseModel):
    chunk: Chunk
    score: float
    vector_score: float | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    title: str
    page: int | None = None
    section: str = ""
    snippet: str
    score: float


class AnswerStatus(str, Enum):
    answered = "answered"
    no_answer = "no_answer"
    no_documents = "no_documents"
    provider_unavailable = "provider_unavailable"
    retrieval_unavailable = "retrieval_unavailable"
    rate_limited = "rate_limited"


class ProviderInfo(BaseModel):
    name: str
    model: str | None = None
    elapsed_ms: int = 0


class Answer(BaseModel):
    """The answer contract. Every field is always present."""

    id: str
    question: str
    collection_id: str
    status: AnswerStatus
    answer: str | None = None
    message: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    passages_considered: int = 0
    provider: ProviderInfo | None = None
    knowledge_version: str = ""
    language: str | None = None
    created_at: datetime


class Job(BaseModel):
    id: str
    collection_id: str
    doc_id: str
    status: DocumentStatus
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class Feedback(BaseModel):
    answer_id: str
    rating: Literal["up", "down"]
    note: str = ""
    created_at: datetime


class ProviderStatus(BaseModel):
    name: str
    installed: bool
    signed_in: bool
    version: str | None = None
    detail: str = ""
    fix_command: str | None = None
    models: list[str] = Field(default_factory=list)
