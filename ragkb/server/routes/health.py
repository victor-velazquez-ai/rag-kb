"""GET /api/health: status, version, knowledge version, counts and store health.

It answers on any Host so uptime checks work through a tunnel, but a caller that is not on this
machine gets the public view: nothing that names a local folder, a user or a program path.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ragkb.server.deps import DepsDep
from ragkb.server.loopback import host_is_loopback

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health(request: Request, deps: DepsDep) -> dict[str, Any]:
    return deps.health(full=host_is_loopback(request.headers.get("host")))
