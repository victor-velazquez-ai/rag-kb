"""GET /api/providers: which answer providers are installed and signed in, with the fix for each."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ragkb.models import ProviderStatus
from ragkb.server.deps import DepsDep
from ragkb.server.errors import ApiError

router = APIRouter(tags=["providers"])


@router.get("/providers", response_model=list[ProviderStatus])
def list_providers(
    deps: DepsDep,
    refresh: Annotated[bool, Query(description="Check again instead of using the short cache.")] = False,
) -> list[ProviderStatus]:
    try:
        return deps.provider_statuses(refresh=refresh)
    except ApiError:
        raise
    except Exception:  # noqa: BLE001 - a status check failure is reported, not raised
        raise ApiError(503, "provider_unavailable", "Provider status could not be checked right now.") from None
