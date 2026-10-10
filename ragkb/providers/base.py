"""Provider interface: anything that turns (system, prompt) into text.

Adapters wrap the vendors' own command-line tools so the answer runs on the user's subscription.
"""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ragkb.models import ProviderStatus

ProviderErrorKind = Literal["not_installed", "not_signed_in", "timeout", "rate_limited", "failed"]


class ProviderError(Exception):
    def __init__(self, kind: ProviderErrorKind, message: str, *, fix_command: str | None = None):
        super().__init__(message)
        self.kind: ProviderErrorKind = kind
        self.message = message
        self.fix_command = fix_command


class ProviderResult(BaseModel):
    text: str
    model: str | None = None
    elapsed_ms: int = 0
    raw: dict = Field(default_factory=dict)


class Provider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult: ...
