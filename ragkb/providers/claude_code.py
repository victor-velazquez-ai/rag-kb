"""Claude Code adapter: answers run on the user's own Claude subscription through `claude -p`.

This module also holds the small process helpers both CLI adapters share: binary discovery, a runner
that sends the prompt on stdin and kills the whole process tree on timeout, flag discovery from
`--help`, and error classification. Prompts and passages are never logged.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkb.config import Settings
from ragkb.models import ProviderStatus
from ragkb.providers.base import ProviderError, ProviderErrorKind, ProviderResult

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"
STATUS_TIMEOUT_S = 20
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Model names reach a command line (and, through an npm .cmd shim, cmd.exe), so only plain names pass.
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\[\]-]{0,99}$")
VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.]+)?")

# Variables a parent Claude Code session sets for its own children. Passing them on would make the
# answer call think it runs inside that session.
NESTED_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING",
    "CLAUDE_AGENT_SDK_VERSION",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
)
# An API key in the environment would bill an API account instead of the user's subscription.
SUBSCRIPTION_ONLY_DROP = ("ANTHROPIC_API_KEY",)


# ---------------------------------------------------------------------------------------------------
# Shared CLI helpers
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CliRun:
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_ms: int
    timed_out: bool = False


@dataclass(frozen=True)
class ExtensionSpec:
    """A binary shipped inside an editor extension: <root>/<prefix><version>-<platform>/<relative>."""

    prefix: str
    relative: str


def child_env(drop: Iterable[str] = ()) -> dict[str, str]:
    env = dict(os.environ)
    for key in (*NESTED_SESSION_VARS, *drop):
        env.pop(key, None)
    return env


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", text)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def editor_extension_roots(home: Path) -> list[Path]:
    return [home / ".vscode" / "extensions", home / ".vscode-insiders" / "extensions"]


def find_newest_in_extensions(home: Path, spec: ExtensionSpec) -> str | None:
    found: list[tuple[tuple[int, ...], str]] = []
    for root in editor_extension_roots(home):
        if not root.is_dir():
            continue
        for candidate in root.glob(f"{spec.prefix}*/{spec.relative}"):
            if not candidate.is_file():
                continue
            folder = next((p for p in candidate.parts if p.startswith(spec.prefix)), "")
            found.append((_version_tuple(folder[len(spec.prefix):]), str(candidate)))
    if not found:
        return None
    found.sort(key=lambda item: item[0], reverse=True)
    return found[0][1]


def find_binary(
    explicit: str,
    *,
    setting_name: str,
    command: str,
    known_dirs: Sequence[Path],
    extension: ExtensionSpec,
) -> tuple[str | None, str]:
    """Locate a CLI. Order: the explicit setting, PATH, known install folders, editor extensions
    (newest version first). Returns (path or None, a short note on how it was found or why not)."""
    explicit = (explicit or "").strip()
    if explicit:
        if Path(explicit).is_file():
            return explicit, f"set by {setting_name}"
        resolved = shutil.which(explicit)
        if resolved:
            return resolved, f"set by {setting_name}"
        return None, f"{setting_name} is set but does not point to a program"
    on_path = shutil.which(command)
    if on_path:
        return on_path, "found on PATH"
    names = [f"{command}.exe", f"{command}.cmd"] if IS_WINDOWS else [command]
    for folder in known_dirs:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return str(candidate), "found in a standard install folder"
    in_extension = find_newest_in_extensions(Path.home(), extension)
    if in_extension:
        return in_extension, "found in the VS Code extension"
    return None, "not found on PATH or in the usual install folders"


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the process and its children: an npm .cmd shim or a CLI that spawns helpers would otherwise
    keep the pipes open and turn the timeout into a hang."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=15, check=False, creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_cli(
    cmd: Sequence[str],
    *,
    input_text: str | None,
    timeout_s: float,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> CliRun:
    """Run *cmd* with *input_text* on stdin as UTF-8. Raises OSError when the program cannot start.

    The pipes are binary on purpose: a text-mode pipe on Windows rewrites every newline of the prompt
    as CRLF. Output is decoded as UTF-8 with replacement and its line endings normalized."""
    started = time.monotonic()
    platform_kwargs: dict[str, object] = (
        {"creationflags": _NO_WINDOW} if IS_WINDOWS else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        **platform_kwargs,  # type: ignore[arg-type]
    )
    payload = (input_text or "").encode("utf-8")
    try:
        stdout, stderr = proc.communicate(payload, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            stdout, stderr = b"", b""
        return CliRun(proc.returncode, _decode(stdout), _decode(stderr), _ms_since(started), timed_out=True)
    return CliRun(proc.returncode, _decode(stdout), _decode(stderr), _ms_since(started))


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace").replace("\r\n", "\n")


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


_HELP_CACHE: dict[str, frozenset[str]] = {}
_HELP_LOCK = threading.Lock()


def help_flags(binary: str, env: dict[str, str] | None = None) -> frozenset[str]:
    """The long options a CLI lists in `--help`, cached per binary. Empty when help cannot be read, so
    optional flags are simply left out rather than risking an "unknown option" failure."""
    try:
        key = f"{binary}|{os.path.getmtime(binary)}"
    except OSError:
        key = binary
    with _HELP_LOCK:
        if key in _HELP_CACHE:
            return _HELP_CACHE[key]
    try:
        run = run_cli([binary, "--help"], input_text=None, timeout_s=STATUS_TIMEOUT_S, env=env)
        flags = frozenset(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", run.stdout + run.stderr))
    except OSError:
        flags = frozenset()
    with _HELP_LOCK:
        _HELP_CACHE[key] = flags
    return flags


_RATE_LIMITED_RE = re.compile(
    r"usage limit|rate[ _-]?limit|limit reached|hit your (?:usage )?limit|too many requests|\b429\b"
    r"|quota|credit balance",
    re.IGNORECASE,
)
_SIGNED_OUT_RE = re.compile(
    r"not logged in|please run /login|log ?in required|please (?:log|sign) ?in|invalid api key"
    r"|oauth token (?:has )?expired|authentication_error|\b401\b|unauthorized|missing bearer",
    re.IGNORECASE,
)


def classify_error(text: str) -> ProviderErrorKind:
    """Map a CLI's error output to a ProviderError kind. Only call it on error output, never on an
    answer (an answer may well mention rate limits)."""
    if _RATE_LIMITED_RE.search(text):
        return "rate_limited"
    if _SIGNED_OUT_RE.search(text):
        return "not_signed_in"
    return "failed"


_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z?\s+")
_NOISE = re.compile(r",?\s*(?:url|cf-ray|request id):\s*\S+", re.IGNORECASE)
_WIN_PATH = re.compile(r"[A-Za-z]:\\\S+")


def diagnostic_text(output: str, prompt: str) -> str:
    """CLI output with every line of our own prompt removed. Some CLIs echo the prompt; passage text
    must never be classified as an error, logged or shown as one."""
    prompt_lines = {line.strip() for line in (prompt or "").splitlines() if line.strip()}
    kept = [line for line in (output or "").splitlines() if line.strip() and line.strip() not in prompt_lines]
    return "\n".join(kept)


def vendor_message(text: str, limit: int = 240) -> str:
    """The most useful error line of a CLI's output, cleaned for display: no timestamps, URLs, request
    ids or local paths."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    errors = [line for line in lines if re.search(r"\berror\b|not logged in|limit", line, re.IGNORECASE)]
    line = (errors or lines or [""])[-1]
    line = _TIMESTAMP_PREFIX.sub("", line)
    line = re.sub(r"^(?:ERROR|Error|error)(?:\s+[\w:]+)?:\s*", "", line)
    line = _NOISE.sub("", line)
    line = _WIN_PATH.sub("<path>", line).strip()
    if len(line) > limit:
        line = line[: limit - 3].rstrip() + "..."
    return line


def check_model_name(model: str, display_name: str) -> None:
    if not MODEL_NAME_RE.match(model):
        raise ProviderError("failed", f"'{model[:40]}' is not a valid model name for {display_name}.")


def parse_version(text: str) -> str | None:
    match = VERSION_RE.search(text or "")
    return match.group(0) if match else None


# ---------------------------------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------------------------------

CLAUDE_MODELS = ["sonnet", "opus", "haiku"]
CLAUDE_SIGN_IN = "claude auth login"
CLAUDE_INSTALL = (
    "irm https://claude.ai/install.ps1 | iex" if IS_WINDOWS else "curl -fsSL https://claude.ai/install.sh | bash"
)
CLAUDE_EXTENSION = ExtensionSpec(
    "anthropic.claude-code-",
    "resources/native-binary/claude.exe" if IS_WINDOWS else "resources/native-binary/claude",
)
# Flags that keep the call a single plain completion. Each is passed only when this CLI's --help
# lists it: `--tools ""` removes every built-in tool, `--safe-mode` skips the user's CLAUDE.md, hooks,
# plugins and MCP servers (sign-in still works), `--strict-mcp-config` loads no MCP server.
CLAUDE_OPTIONAL_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("--tools", ("--tools", "")),
    ("--safe-mode", ("--safe-mode",)),
    ("--strict-mcp-config", ("--strict-mcp-config",)),
)


def _claude_known_dirs() -> list[Path]:
    home = Path.home()
    dirs = [home / ".local" / "bin", home / ".claude" / "local"]
    if IS_WINDOWS:
        appdata = os.environ.get("APPDATA")
        if appdata:
            dirs.append(Path(appdata) / "npm")
    else:
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), home / ".npm-global" / "bin"]
    return dirs


def _json_payload(stdout: str) -> dict | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def model_from_usage(model_usage: object) -> str | None:
    """The answering model from `modelUsage`: the first entry that is not Haiku (the CLI's helper)."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    names = [str(name) for name in model_usage]
    return next((name for name in names if "haiku" not in name.lower()), names[0])


class ClaudeCodeProvider:
    name = "claude_code"
    display_name = "Claude Code"
    install_command = CLAUDE_INSTALL
    sign_in_command = CLAUDE_SIGN_IN

    def __init__(self, settings: Settings):
        self._settings = settings

    # -- discovery --------------------------------------------------------------------------------

    def locate(self) -> tuple[str | None, str]:
        return find_binary(
            self._settings.CLAUDE_BIN,
            setting_name="RAGKB_CLAUDE_BIN",
            command="claude",
            known_dirs=_claude_known_dirs(),
            extension=CLAUDE_EXTENSION,
        )

    def _env(self) -> dict[str, str]:
        return child_env(SUBSCRIPTION_ONLY_DROP)

    @staticmethod
    def _workdir() -> Path:
        """A stable, empty working folder, so no project file or instruction file is picked up and the
        CLI does not record a new project per call."""
        path = Path(tempfile.gettempdir()) / "ragkb-claude-code"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def status(self) -> ProviderStatus:
        binary, how = self.locate()
        if not binary:
            return ProviderStatus(
                name=self.name, installed=False, signed_in=False, version=None,
                detail=f"Claude Code is not installed ({how}).", fix_command=self.install_command,
                models=list(CLAUDE_MODELS),
            )
        env = self._env()
        version: str | None = None
        try:
            run = run_cli([binary, "--version"], input_text=None, timeout_s=STATUS_TIMEOUT_S, env=env)
            version = parse_version(run.stdout or run.stderr)
            auth = run_cli([binary, "auth", "status"], input_text=None, timeout_s=STATUS_TIMEOUT_S, env=env)
        except OSError:
            return ProviderStatus(
                name=self.name, installed=False, signed_in=False, version=None,
                detail="Claude Code was found but could not be started.", fix_command=self.install_command,
                models=list(CLAUDE_MODELS),
            )
        payload = _json_payload(auth.stdout)
        if auth.timed_out or payload is None:
            return ProviderStatus(
                name=self.name, installed=True, signed_in=False, version=version,
                detail="Could not read the Claude Code sign-in status.", fix_command=self.sign_in_command,
                models=list(CLAUDE_MODELS),
            )
        signed_in = bool(payload.get("loggedIn"))
        if signed_in:
            plan = payload.get("subscriptionType")
            method = payload.get("authMethod") or "an account"
            detail = f"Signed in with {method}" + (f" ({plan} plan)." if plan else ".")
        else:
            detail = "Installed but not signed in."
        return ProviderStatus(
            name=self.name, installed=True, signed_in=signed_in, version=version, detail=detail,
            fix_command=None if signed_in else self.sign_in_command, models=list(CLAUDE_MODELS),
        )

    # -- generation -------------------------------------------------------------------------------

    def build_command(self, binary: str, model: str, system_prompt_file: str) -> list[str]:
        """The headless call. The prompt goes on stdin; the system prompt goes in a file because a
        multi-line argument does not survive an npm .cmd shim on Windows. Never `--bare`: it skips
        the stored sign-in."""
        cmd = [
            binary, "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--model", model,
            "--system-prompt-file", system_prompt_file,
        ]
        available = help_flags(binary, self._env())
        for flag, args in CLAUDE_OPTIONAL_FLAGS:
            if flag in available:
                cmd.extend(args)
        return cmd

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult:
        binary, how = self.locate()
        if not binary:
            raise ProviderError(
                "not_installed", f"Claude Code is not installed ({how}).", fix_command=self.install_command
            )
        model_name = (model or self._settings.CLAUDE_MODEL or "sonnet").strip()
        check_model_name(model_name, self.display_name)
        workdir = self._workdir()
        descriptor, system_path = tempfile.mkstemp(suffix=".txt", prefix="system-", dir=workdir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(system.encode("utf-8"))
            cmd = self.build_command(binary, model_name, system_path)
            run = run_cli(cmd, input_text=prompt, timeout_s=timeout_s, cwd=workdir, env=self._env())
        except OSError as exc:
            raise ProviderError(
                "not_installed", f"Claude Code could not be started ({type(exc).__name__}).",
                fix_command=self.install_command,
            ) from exc
        finally:
            try:
                os.unlink(system_path)
            except OSError:
                pass
        if run.timed_out:
            raise ProviderError("timeout", f"Claude Code did not answer within {timeout_s} seconds.")
        return self.parse_run(run, prompt, model_name)

    def parse_run(self, run: CliRun, prompt: str, requested_model: str) -> ProviderResult:
        payload = _json_payload(run.stdout)
        if payload is None:
            detail = diagnostic_text(f"{run.stderr}\n{run.stdout}", prompt)
            raise self._error(classify_error(detail), vendor_message(detail) or "no output")
        result = payload.get("result")
        if payload.get("is_error") or not isinstance(result, str):
            detail = result if isinstance(result, str) else ""
            status = payload.get("api_error_status")
            if status in (401, 403):
                kind: ProviderErrorKind = "not_signed_in"
            elif status == 429:
                kind = "rate_limited"
            else:
                kind = classify_error(detail)
            raise self._error(kind, vendor_message(detail) or f"error (status {status})")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        raw = {
            "duration_ms": payload.get("duration_ms"),
            "duration_api_ms": payload.get("duration_api_ms"),
            "num_turns": payload.get("num_turns"),
            "total_cost_usd": payload.get("total_cost_usd"),
            "models": list(payload.get("modelUsage") or {}),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        return ProviderResult(
            text=result.strip(),
            model=model_from_usage(payload.get("modelUsage")) or requested_model,
            elapsed_ms=run.elapsed_ms,
            raw=raw,
        )

    def _error(self, kind: ProviderErrorKind, detail: str) -> ProviderError:
        logger.warning("claude_code call failed: kind=%s detail=%s", kind, detail)
        if kind == "not_signed_in":
            return ProviderError(kind, "Claude Code is not signed in.", fix_command=self.sign_in_command)
        if kind == "rate_limited":
            return ProviderError(kind, detail or "Claude Code reached its usage limit.")
        return ProviderError(kind, f"Claude Code failed: {detail}")
