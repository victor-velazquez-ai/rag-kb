"""Codex adapter: answers run on the user's own ChatGPT subscription through `codex exec`.

Codex has no system-prompt flag, so the instructions travel in the same stdin document as the task.
The answer is read from the file given to `-o`; the console output only feeds diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

from ragkb.answer.prompts import AGENT_PROMPT_TEMPLATE
from ragkb.config import Settings
from ragkb.models import ProviderStatus
from ragkb.providers.base import ProviderError, ProviderErrorKind, ProviderResult
from ragkb.providers.claude_code import (
    IS_WINDOWS,
    STATUS_TIMEOUT_S,
    CliRun,
    ExtensionSpec,
    check_model_name,
    child_env,
    classify_error,
    diagnostic_text,
    find_binary,
    parse_version,
    run_cli,
    vendor_message,
)

logger = logging.getLogger(__name__)

CODEX_SIGN_IN = "codex login"
# An API key in the environment would bill an API account instead of the user's ChatGPT plan.
SUBSCRIPTION_ONLY_DROP = ("OPENAI_API_KEY",)
CODEX_INSTALL = "npm install -g @openai/codex"
CODEX_EXTENSION = ExtensionSpec(
    "openai.chatgpt-",
    "bin/*/codex.exe" if IS_WINDOWS else "bin/*/codex",
)
REASONING_RE = re.compile(r"^[a-z]{2,10}$")
_TOKENS_RE = re.compile(r"tokens used\s*:?\s*\n?\s*([\d,]+)", re.IGNORECASE)
_MODEL_LINE_RE = re.compile(r"^model:\s*(\S+)\s*$", re.MULTILINE)


def _codex_known_dirs() -> list[Path]:
    home = Path.home()
    dirs = [home / ".local" / "bin"]
    if IS_WINDOWS:
        appdata = os.environ.get("APPDATA")
        if appdata:
            dirs.append(Path(appdata) / "npm")
    else:
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), home / ".npm-global" / "bin"]
    return dirs


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def configured_model(home: Path | None = None) -> str | None:
    """The `model` set in the user's Codex config.toml, if any."""
    path = (home or codex_home()) / "config.toml"
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return value if isinstance(value, str) and value.strip() else None


def known_models(home: Path | None = None) -> list[str]:
    """Models this Codex install has confirmed: the configured model first, then the listed entries of
    the CLI's own models cache in its priority order. Nothing is invented when neither exists."""
    home = home or codex_home()
    models: list[str] = []
    configured = configured_model(home)
    if configured:
        models.append(configured)
    try:
        cache = json.loads((home / "models_cache.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    entries = cache.get("models") if isinstance(cache, dict) else None
    if isinstance(entries, list):
        listed = [
            entry for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("slug"), str) and entry.get("visibility") == "list"
        ]
        listed.sort(key=lambda entry: entry.get("priority") if isinstance(entry.get("priority"), int) else 999)
        for entry in listed:
            if entry["slug"] not in models:
                models.append(entry["slug"])
    return models


def parse_tokens_used(text: str) -> int | None:
    match = _TOKENS_RE.search(text or "")
    return int(match.group(1).replace(",", "")) if match else None


class CodexProvider:
    name = "codex"
    display_name = "Codex"
    install_command = CODEX_INSTALL
    sign_in_command = CODEX_SIGN_IN

    def __init__(self, settings: Settings):
        self._settings = settings

    def locate(self) -> tuple[str | None, str]:
        return find_binary(
            self._settings.CODEX_BIN,
            setting_name="RAGKB_CODEX_BIN",
            command="codex",
            known_dirs=_codex_known_dirs(),
            extension=CODEX_EXTENSION,
        )

    def status(self) -> ProviderStatus:
        binary, how = self.locate()
        models = known_models()
        if not binary:
            return ProviderStatus(
                name=self.name, installed=False, signed_in=False, version=None,
                detail=f"Codex is not installed ({how}).", fix_command=self.install_command, models=models,
            )
        env = child_env(SUBSCRIPTION_ONLY_DROP)
        try:
            run = run_cli([binary, "--version"], input_text=None, timeout_s=STATUS_TIMEOUT_S, env=env)
            version = parse_version(run.stdout or run.stderr)
            login = run_cli([binary, "login", "status"], input_text=None, timeout_s=STATUS_TIMEOUT_S, env=env)
        except OSError:
            return ProviderStatus(
                name=self.name, installed=False, signed_in=False, version=None,
                detail="Codex was found but could not be started.", fix_command=self.install_command,
                models=models,
            )
        text = f"{login.stdout}\n{login.stderr}".lower()
        if "not logged in" in text or login.timed_out:
            signed_in, detail = False, "Installed but not signed in."
        elif "logged in using chatgpt" in text:
            signed_in, detail = True, "Signed in with ChatGPT."
        elif "logged in using an api key" in text or "logged in using api key" in text:
            signed_in, detail = True, "Signed in with an API key, so usage is billed to that API account."
        elif "logged in" in text and login.returncode == 0:
            signed_in, detail = True, "Signed in."
        else:
            signed_in, detail = False, "Could not read the Codex sign-in status."
        return ProviderStatus(
            name=self.name, installed=True, signed_in=signed_in, version=version, detail=detail,
            fix_command=None if signed_in else self.sign_in_command, models=models,
        )

    # -- generation -------------------------------------------------------------------------------

    def build_command(self, binary: str, workdir: str, outfile: str, model: str | None) -> list[str]:
        """The headless call: no git check, nothing persisted, read-only sandbox in an empty folder, the
        last message written to *outfile*. No positional prompt, so Codex reads it from stdin."""
        cmd = [
            binary, "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s", "read-only",
            "-C", workdir,
            "-o", outfile,
            "--color", "never",
        ]
        if model:
            cmd += ["-m", model]
        effort = (self._settings.CODEX_REASONING or "").strip().lower()
        if REASONING_RE.match(effort):
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        return cmd

    @staticmethod
    def wrap_prompt(system: str, prompt: str) -> str:
        return AGENT_PROMPT_TEMPLATE.format(system=system.strip() + "\n", prompt=prompt)

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult:
        binary, how = self.locate()
        if not binary:
            raise ProviderError("not_installed", f"Codex is not installed ({how}).", fix_command=self.install_command)
        model_name = (model or self._settings.CODEX_MODEL or "").strip() or None
        if model_name:
            check_model_name(model_name, self.display_name)
        workdir = tempfile.mkdtemp(prefix="ragkb-codex-")
        outfile = os.path.join(workdir, "answer.md")
        full_prompt = self.wrap_prompt(system, prompt)
        try:
            cmd = self.build_command(binary, workdir, outfile, model_name)
            run = run_cli(cmd, input_text=full_prompt, timeout_s=timeout_s, cwd=workdir, env=child_env(SUBSCRIPTION_ONLY_DROP))
            if run.timed_out:
                raise ProviderError("timeout", f"Codex did not answer within {timeout_s} seconds.")
            answer = self._read_answer(outfile)
            return self.parse_run(run, answer, full_prompt, model_name)
        except OSError as exc:
            raise ProviderError(
                "not_installed", f"Codex could not be started ({type(exc).__name__}).",
                fix_command=self.install_command,
            ) from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _read_answer(outfile: str) -> str | None:
        try:
            return Path(outfile).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def parse_run(self, run: CliRun, answer: str | None, prompt: str, requested_model: str | None) -> ProviderResult:
        diagnostics = diagnostic_text(f"{run.stdout}\n{run.stderr}", prompt)
        if run.returncode != 0 or answer is None:
            raise self._error(classify_error(diagnostics), vendor_message(diagnostics) or "no answer")
        model_match = _MODEL_LINE_RE.search(diagnostics)
        model = model_match.group(1) if model_match else (requested_model or configured_model())
        return ProviderResult(
            text=answer.strip(),
            model=model,
            elapsed_ms=run.elapsed_ms,
            raw={"tokens_used": parse_tokens_used(diagnostics), "returncode": run.returncode},
        )

    def _error(self, kind: ProviderErrorKind, detail: str) -> ProviderError:
        logger.warning("codex call failed: kind=%s detail=%s", kind, detail)
        if kind == "not_signed_in":
            return ProviderError(kind, "Codex is not signed in.", fix_command=self.sign_in_command)
        if kind == "rate_limited":
            return ProviderError(kind, detail or "Codex reached its usage limit.")
        return ProviderError(kind, f"Codex failed: {detail}")
