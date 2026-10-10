"""Provider detection and selection for the API, the UI and `ragkb doctor`."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ragkb.config import Settings
from ragkb.models import ProviderStatus
from ragkb.providers.base import Provider, ProviderError
from ragkb.providers.claude_code import ClaudeCodeProvider
from ragkb.providers.codex import CodexProvider
from ragkb.providers.extractive import ExtractiveProvider

logger = logging.getLogger(__name__)

PROVIDER_ORDER = ("claude_code", "codex", "extractive")
AUTO = "auto"
STATUS_TTL_S = 30.0


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._providers: dict[str, Provider] = {
            "claude_code": ClaudeCodeProvider(settings),
            "codex": CodexProvider(settings),
            "extractive": ExtractiveProvider(),
        }
        self._lock = threading.Lock()
        self._cached: tuple[float, list[ProviderStatus]] | None = None

    # -- status -----------------------------------------------------------------------------------

    def statuses(self, *, refresh: bool = False) -> list[ProviderStatus]:
        """Status of claude_code, codex and extractive, in that order. Each check runs the CLIs, so the
        result is cached for STATUS_TTL_S seconds; the two CLI checks run in parallel."""
        with self._lock:
            if not refresh and self._cached and time.monotonic() - self._cached[0] < STATUS_TTL_S:
                return [status.model_copy() for status in self._cached[1]]
        with ThreadPoolExecutor(max_workers=len(PROVIDER_ORDER)) as pool:
            futures = [pool.submit(self._safe_status, name) for name in PROVIDER_ORDER]
            result = [future.result() for future in futures]
        with self._lock:
            self._cached = (time.monotonic(), result)
        return [status.model_copy() for status in result]

    def _safe_status(self, name: str) -> ProviderStatus:
        try:
            return self._providers[name].status()
        except Exception as exc:  # noqa: BLE001 - a broken CLI must not break the health page
            logger.warning("provider status check failed: provider=%s error=%s", name, type(exc).__name__)
            return ProviderStatus(
                name=name, installed=False, signed_in=False, version=None,
                detail="The status check failed.", fix_command=None, models=[],
            )

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None

    def status_of(self, name: str) -> ProviderStatus:
        for status in self.statuses():
            if status.name == name:
                return status
        raise ValueError(f"Unknown provider '{name}'.")

    def get(self, name: str) -> Provider:
        """The provider object by name, without any availability check."""
        try:
            return self._providers[name]
        except KeyError:
            raise ValueError(self._unknown(name)) from None

    # -- selection --------------------------------------------------------------------------------

    def select(self, name: str | None = None) -> Provider:
        """`None` means the configured default (RAGKB_PROVIDER, `auto` unless set). `auto` picks the first
        signed-in of claude_code and codex, else extractive. A named provider that is not usable raises
        ProviderError(not_installed | not_signed_in) with its fix command; an unknown name raises
        ValueError."""
        wanted = (name or self._settings.PROVIDER or AUTO).strip().lower()
        if wanted == AUTO:
            provider, reason = self._auto()
            logger.info("provider selected: %s (%s)", provider.name, reason)
            return provider
        if wanted not in self._providers:
            raise ValueError(self._unknown(wanted))
        provider = self._providers[wanted]
        status = self.status_of(wanted)
        if not status.installed:
            raise ProviderError(
                "not_installed", f"{_display(provider)} is not installed.", fix_command=status.fix_command
            )
        if not status.signed_in:
            raise ProviderError(
                "not_signed_in", f"{_display(provider)} is not signed in.", fix_command=status.fix_command
            )
        logger.info("provider selected: %s (requested)", wanted)
        return provider

    def _auto(self) -> tuple[Provider, str]:
        statuses = {status.name: status for status in self.statuses()}
        skipped: list[str] = []
        for name in ("claude_code", "codex"):
            status = statuses.get(name)
            if status and status.installed and status.signed_in:
                note = f"auto: first signed-in provider{', ' + '; '.join(skipped) if skipped else ''}"
                return self._providers[name], note
            if status is None or not status.installed:
                skipped.append(f"{name} not installed")
            else:
                skipped.append(f"{name} not signed in")
        return self._providers["extractive"], "auto: " + "; ".join(skipped) + ", using extractive"

    @staticmethod
    def _unknown(name: str) -> str:
        return f"Unknown provider '{name}'. Choose auto, {', '.join(PROVIDER_ORDER)}."

    # -- doctor -----------------------------------------------------------------------------------

    def doctor_report(self) -> dict:
        """Everything `ragkb doctor` prints: per provider whether it is installed and signed in, its
        version, the binary it would run, the fix command, and the provider an answer would use now."""
        statuses = self.statuses(refresh=True)
        providers = []
        for status in statuses:
            provider = self._providers[status.name]
            locate = getattr(provider, "locate", None)
            binary, found = locate() if callable(locate) else (None, "built in")
            providers.append(
                {
                    "name": status.name,
                    "display_name": _display(provider),
                    "installed": status.installed,
                    "signed_in": status.signed_in,
                    "version": status.version,
                    "detail": status.detail,
                    "fix_command": status.fix_command,
                    "models": status.models,
                    "binary": binary,
                    "found": found,
                }
            )
        configured = (self._settings.PROVIDER or AUTO).strip().lower()
        try:
            chosen = self.select(None)
            default: dict = {"name": chosen.name, "ok": True, "message": None, "fix_command": None}
        except ProviderError as exc:
            default = {"name": None, "ok": False, "message": exc.message, "fix_command": exc.fix_command}
        except ValueError as exc:
            default = {"name": None, "ok": False, "message": str(exc), "fix_command": None}
        return {
            "configured": configured,
            "default": default,
            "providers": providers,
            "settings": {
                "claude_model": self._settings.CLAUDE_MODEL,
                "codex_model": self._settings.CODEX_MODEL or None,
                "codex_reasoning": self._settings.CODEX_REASONING,
                "answer_timeout_s": self._settings.ANSWER_TIMEOUT_S,
            },
        }


def _display(provider: object) -> str:
    return str(getattr(provider, "display_name", None) or getattr(provider, "name", "provider"))
