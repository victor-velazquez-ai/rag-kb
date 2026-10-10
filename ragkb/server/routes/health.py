"""GET /api/health: status, version, knowledge version, counts and store health."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ragkb.server.deps import DepsDep

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health(deps: DepsDep) -> dict[str, Any]:
    return deps.health()
