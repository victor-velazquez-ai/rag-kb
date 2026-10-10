"""Provider adapters against fake CLIs that replay recorded outputs, plus discovery and the registry."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ragkb.config import Settings
from ragkb.models import ProviderStatus
from ragkb.providers import claude_code as cc
from ragkb.providers.base import ProviderError, ProviderResult
from ragkb.providers.claude_code import (
    CLAUDE_EXTENSION,
    ClaudeCodeProvider,
    classify_error,
    diagnostic_text,
    find_binary,
    model_from_usage,
    vendor_message,
)
from ragkb.providers.codex import CODEX_EXTENSION, CodexProvider, known_models, parse_tokens_used
from ragkb.providers.extractive import ExtractiveProvider
from ragkb.providers.registry import ProviderRegistry
from tests.fake_clis import FIXTURES, make_shim

SYSTEM = "You are a test system prompt.\nLine two with \"quotes\" and 100% & more."
PROMPT = "Numbered passages:\n\n[1] (Doc, page 1)\nThe answer is 42.\n\nQuestion: What is the answer?"


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_CLI_RECORD", str(path))
    for name in ("FAKE_CLAUDE_SCENARIO", "FAKE_CLAUDE_AUTH", "FAKE_CLAUDE_RESULT",
                 "FAKE_CODEX_SCENARIO", "FAKE_CODEX_AUTH", "FAKE_CODEX_RESULT"):
        monkeypatch.delenv(name, raising=False)
    return path


def recorded(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def claude(tmp_path: Path, record: Path) -> ClaudeCodeProvider:
    shim = make_shim(tmp_path / "bin", "claude", "fake_claude.py")
    return ClaudeCodeProvider(Settings(CLAUDE_BIN=str(shim), CLAUDE_MODEL="sonnet"))


@pytest.fixture
def codex(tmp_path: Path, record: Path) -> CodexProvider:
    shim = make_shim(tmp_path / "bin", "codex", "fake_codex.py")
    return CodexProvider(Settings(CODEX_BIN=str(shim), CODEX_MODEL="", CODEX_REASONING="medium"))


# ---------------------------------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------------------------------

def test_claude_command_prompt_on_stdin_and_single_completion_flags(claude: ClaudeCodeProvider, record: Path):
    result = claude.generate(SYSTEM, PROMPT, timeout_s=30)
    call = recorded(record)
    argv = call["argv"]
    assert argv[:2] == ["-p", "--output-format"] and argv[2] == "json"
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--tools") + 1] == ""
    assert "--safe-mode" in argv and "--strict-mcp-config" in argv
    assert "--bare" not in argv
    assert PROMPT not in argv and call["stdin"] == PROMPT
    assert call["system_prompt"] == SYSTEM
    system_file = Path(argv[argv.index("--system-prompt-file") + 1])
    assert not system_file.exists(), "the system prompt file is removed after the call"
    fixture = json.loads((FIXTURES / "claude_success.json").read_text(encoding="utf-8"))
    assert result.text == fixture["result"]
    assert result.model == "claude-sonnet-5-5"
    assert result.raw["num_turns"] == 1 and result.elapsed_ms > 0


def test_claude_model_override_is_passed(claude: ClaudeCodeProvider, record: Path):
    claude.generate(SYSTEM, PROMPT, timeout_s=30, model="opus")
    argv = recorded(record)["argv"]
    assert argv[argv.index("--model") + 1] == "opus"


def test_claude_optional_flags_left_out_when_help_does_not_list_them(
    claude: ClaudeCodeProvider, record: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "old_help")
    claude.generate(SYSTEM, PROMPT, timeout_s=30)
    argv = recorded(record)["argv"]
    assert "--tools" not in argv and "--safe-mode" not in argv and "--strict-mcp-config" not in argv
    assert "--system-prompt-file" in argv


def test_claude_runs_on_the_subscription_not_an_api_key(
    claude: ClaudeCodeProvider, record: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDECODE", "1")
    claude.generate(SYSTEM, PROMPT, timeout_s=30)
    call = recorded(record)
    assert call["env_anthropic_api_key"] is False
    assert call["env_claudecode"] is False


def test_claude_model_is_the_first_non_haiku_entry(
    claude: ClaudeCodeProvider, record: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "haiku_first")
    assert claude.generate(SYSTEM, PROMPT, timeout_s=30).model == "claude-sonnet-5-5"


def test_model_from_usage():
    assert model_from_usage({"claude-haiku-4-5": {}, "claude-opus-5": {}}) == "claude-opus-5"
    assert model_from_usage({"claude-haiku-4-5": {}}) == "claude-haiku-4-5"
    assert model_from_usage({}) is None
    assert model_from_usage(None) is None


def test_claude_not_logged_in(claude: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "not_logged_in")
    with pytest.raises(ProviderError) as err:
        claude.generate(SYSTEM, PROMPT, timeout_s=30)
    assert err.value.kind == "not_signed_in"
    assert err.value.fix_command == "claude auth login"


def test_claude_rate_limited(claude: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "rate_limited")
    with pytest.raises(ProviderError) as err:
        claude.generate(SYSTEM, PROMPT, timeout_s=30)
    assert err.value.kind == "rate_limited"
    assert "usage limit" in err.value.message


def test_claude_crash_without_json_is_failed(claude: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "crash")
    with pytest.raises(ProviderError) as err:
        claude.generate(SYSTEM, PROMPT, timeout_s=30)
    assert err.value.kind == "failed"
    assert "something broke" in err.value.message


def test_claude_timeout_kills_the_process_tree(claude: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "hang")
    started = time.monotonic()
    with pytest.raises(ProviderError) as err:
        claude.generate(SYSTEM, PROMPT, timeout_s=2)
    assert err.value.kind == "timeout"
    assert time.monotonic() - started < 20, "a hung CLI must not hold the caller past the timeout"


def test_claude_rejects_unsafe_model_names_before_running(claude: ClaudeCodeProvider, record: Path):
    with pytest.raises(ProviderError) as err:
        claude.generate(SYSTEM, PROMPT, timeout_s=30, model='sonnet" & calc & "')
    assert err.value.kind == "failed"
    assert not record.exists()


def test_claude_status_signed_in(claude: ClaudeCodeProvider):
    status = claude.status()
    assert status.installed and status.signed_in
    assert status.version == "2.1.292"
    assert status.fix_command is None
    assert status.models == ["sonnet", "opus", "haiku"]
    assert "max plan" in status.detail
    assert "example.com" not in status.detail


def test_claude_status_signed_out(claude: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", "out")
    status = claude.status()
    assert status.installed and not status.signed_in
    assert status.fix_command == "claude auth login"


def test_claude_not_installed(tmp_path: Path):
    provider = ClaudeCodeProvider(Settings(CLAUDE_BIN=str(tmp_path / "missing" / "claude.exe")))
    status = provider.status()
    assert not status.installed and not status.signed_in
    assert status.fix_command == cc.CLAUDE_INSTALL
    assert "RAGKB_CLAUDE_BIN" in status.detail
    with pytest.raises(ProviderError) as err:
        provider.generate(SYSTEM, PROMPT, timeout_s=5)
    assert err.value.kind == "not_installed"


# ---------------------------------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------------------------------

def test_codex_command_and_stdin(codex: CodexProvider, record: Path):
    result = codex.generate(SYSTEM, PROMPT, timeout_s=30)
    call = recorded(record)
    argv = call["argv"]
    assert argv[:5] == ["exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"]
    workdir = argv[argv.index("-C") + 1]
    outfile = argv[argv.index("-o") + 1]
    assert Path(outfile).parent == Path(workdir)
    assert call["workdir_exists"] and not Path(workdir).exists(), "the temp folder is removed afterwards"
    assert "-m" not in argv, "an empty CODEX_MODEL leaves the CLI's own default"
    assert argv[argv.index("-c") + 1] == "model_reasoning_effort=medium"
    stdin = call["stdin"]
    assert stdin.startswith("# Instructions\n\n" + SYSTEM)
    assert stdin.index("# Instructions") < stdin.index("# Task") < stdin.index(PROMPT)
    assert PROMPT not in argv
    answer = (FIXTURES / "codex_success.answer.md").read_text(encoding="utf-8")
    assert result.text == answer.strip()
    assert result.model == "gpt-6-sol"
    assert result.raw["tokens_used"] == 2756


def test_codex_model_and_reasoning_settings(tmp_path: Path, record: Path):
    shim = make_shim(tmp_path / "bin", "codex", "fake_codex.py")
    provider = CodexProvider(Settings(CODEX_BIN=str(shim), CODEX_MODEL="gpt-6-luna", CODEX_REASONING="low"))
    provider.generate(SYSTEM, PROMPT, timeout_s=30, model="gpt-6-astra")
    argv = recorded(record)["argv"]
    assert argv[argv.index("-m") + 1] == "gpt-6-astra"
    assert argv[argv.index("-c") + 1] == "model_reasoning_effort=low"


def test_codex_drops_openai_api_key(codex: CodexProvider, record: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    codex.generate(SYSTEM, PROMPT, timeout_s=30)
    assert recorded(record)["env_openai_api_key"] is False


def test_codex_not_logged_in(codex: CodexProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "not_logged_in")
    with pytest.raises(ProviderError) as err:
        codex.generate(SYSTEM, PROMPT, timeout_s=30)
    assert err.value.kind == "not_signed_in"
    assert err.value.fix_command == "codex login"


def test_codex_rate_limited_keeps_the_vendor_message(codex: CodexProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "rate_limited")
    with pytest.raises(ProviderError) as err:
        codex.generate(SYSTEM, PROMPT, timeout_s=30)
    assert err.value.kind == "rate_limited"
    assert "try again in 2 hours" in err.value.message


def test_codex_timeout(codex: CodexProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "hang")
    started = time.monotonic()
    with pytest.raises(ProviderError) as err:
        codex.generate(SYSTEM, PROMPT, timeout_s=2)
    assert err.value.kind == "timeout"
    assert time.monotonic() - started < 20


def test_codex_echoed_prompt_is_not_read_as_the_error(codex: CodexProvider, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "echo_error")
    passages = (
        "Numbered passages:\n\n[1] (Ops guide)\nERROR: rate limit exceeded means the API returned 429.\n"
        "A 401 Unauthorized reply means you are not logged in.\n\nQuestion: What does 429 mean?"
    )
    with pytest.raises(ProviderError) as err:
        codex.generate(SYSTEM, passages, timeout_s=30)
    assert err.value.kind == "failed"
    assert "stream disconnected" in err.value.message
    assert "429" not in err.value.message and "Ops guide" not in err.value.message


def test_codex_status_signed_in_and_out(codex: CodexProvider, monkeypatch: pytest.MonkeyPatch):
    status = codex.status()
    assert status.installed and status.signed_in and status.version == "0.162.0-alpha.2"
    assert status.detail == "Signed in with ChatGPT."
    monkeypatch.setenv("FAKE_CODEX_AUTH", "out")
    status = codex.status()
    assert status.installed and not status.signed_in and status.fix_command == "codex login"


def test_codex_models_come_from_the_cli_config_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('model = "model-b"\nmodel_reasoning_effort = "high"\n', encoding="utf-8")
    cache = {"models": [
        {"slug": "model-c", "visibility": "list", "priority": 9},
        {"slug": "model-a", "visibility": "list", "priority": 2},
        {"slug": "model-b", "visibility": "list", "priority": 3},
        {"slug": "hidden-model", "visibility": "hide", "priority": 1},
    ]}
    (home / "models_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    assert known_models() == ["model-b", "model-a", "model-c"]
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty"))
    assert known_models() == []


def test_parse_tokens_used():
    stderr = (FIXTURES / "codex_success.stderr.txt").read_text(encoding="utf-8")
    assert parse_tokens_used(stderr) == 2756
    assert parse_tokens_used("tokens used: 1,234") == 1234
    assert parse_tokens_used("no count here") is None


# ---------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Not logged in \u00b7 Please run /login", "not_signed_in"),
        ("unexpected status 401 Unauthorized: Missing bearer", "not_signed_in"),
        ("OAuth token has expired", "not_signed_in"),
        ("You've hit your usage limit \u00b7 resets 9pm", "rate_limited"),
        ("Claude AI usage limit reached|1760000000", "rate_limited"),
        ("429 Too Many Requests", "rate_limited"),
        ("insufficient_quota", "rate_limited"),
        ("stream disconnected before completion", "failed"),
    ],
)
def test_classify_error(text: str, kind: str):
    assert classify_error(text) == kind


def test_vendor_message_is_cleaned():
    text = (FIXTURES / "codex_not_logged_in.stderr.txt").read_text(encoding="utf-8")
    message = vendor_message(text)
    assert message.startswith("unexpected status 401 Unauthorized")
    assert "https://" not in message and "cf-ray" not in message and "request id" not in message
    assert vendor_message(r"Error: cannot open C:\Users\someone\secret\file.txt now") == "cannot open <path> now"


def test_diagnostic_text_removes_prompt_lines():
    output = "header\nThe answer is 42.\nERROR: boom"
    assert diagnostic_text(output, PROMPT) == "header\nERROR: boom"


def _isolate_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return home


def _make_extension(home: Path, spec, folder: str) -> Path:
    relative = spec.relative.replace("*", "windows-x86_64" if os.name == "nt" else "linux-x86_64")
    path = home / ".vscode" / "extensions" / folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_discovery_picks_the_newest_claude_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_discovery(tmp_path, monkeypatch)
    _make_extension(home, CLAUDE_EXTENSION, "anthropic.claude-code-2.1.9-win32-x64")
    newest = _make_extension(home, CLAUDE_EXTENSION, "anthropic.claude-code-2.1.292-win32-x64")
    _make_extension(home, CLAUDE_EXTENSION, "anthropic.claude-code-2.1.30-win32-x64")
    path, how = ClaudeCodeProvider(Settings(CLAUDE_BIN="")).locate()
    assert path == str(newest)
    assert "extension" in how


def test_discovery_picks_the_newest_codex_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_discovery(tmp_path, monkeypatch)
    _make_extension(home, CODEX_EXTENSION, "openai.chatgpt-26.999.1-win32-x64")
    newest = _make_extension(home, CODEX_EXTENSION, "openai.chatgpt-26.1002.51308-win32-x64")
    path, _ = CodexProvider(Settings(CODEX_BIN="")).locate()
    assert path == str(newest)


def test_discovery_order_explicit_then_path_then_install_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_discovery(tmp_path, monkeypatch)
    _make_extension(home, CLAUDE_EXTENSION, "anthropic.claude-code-2.1.292-win32-x64")
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    installed = local_bin / ("claude.exe" if os.name == "nt" else "claude")
    installed.write_bytes(b"")
    assert ClaudeCodeProvider(Settings(CLAUDE_BIN="")).locate()[0] == str(installed)

    on_path = make_shim(tmp_path / "on-path", "claude", "fake_claude.py")
    monkeypatch.setenv("PATH", str(on_path.parent))
    found, how = ClaudeCodeProvider(Settings(CLAUDE_BIN="")).locate()
    assert Path(found).resolve() == on_path.resolve() and how == "found on PATH"

    explicit = make_shim(tmp_path / "explicit", "claude", "fake_claude.py")
    found, how = ClaudeCodeProvider(Settings(CLAUDE_BIN=str(explicit))).locate()
    assert found == str(explicit) and "RAGKB_CLAUDE_BIN" in how


def test_discovery_explicit_setting_that_points_nowhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_discovery(tmp_path, monkeypatch)
    path, how = find_binary(
        str(tmp_path / "nope.exe"), setting_name="RAGKB_CLAUDE_BIN", command="claude",
        known_dirs=[], extension=CLAUDE_EXTENSION,
    )
    assert path is None and "RAGKB_CLAUDE_BIN" in how


# ---------------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------------

class StubProvider:
    def __init__(self, name: str, *, installed: bool = True, signed_in: bool = True, fix: str | None = None):
        self.name = name
        self.display_name = name.title()
        self.installed = installed
        self.signed_in = signed_in
        self.fix = fix
        self.status_calls = 0

    def status(self) -> ProviderStatus:
        self.status_calls += 1
        return ProviderStatus(name=self.name, installed=self.installed, signed_in=self.signed_in,
                              version="1.0", detail="stub", fix_command=self.fix, models=[])

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult:
        return ProviderResult(text="stub")


def _registry(settings: Settings | None = None, **stubs: StubProvider) -> ProviderRegistry:
    registry = ProviderRegistry(settings or Settings(PROVIDER="auto"))
    for name, stub in stubs.items():
        registry._providers[name] = stub
    return registry


def test_registry_auto_prefers_claude_then_codex_then_extractive():
    registry = _registry(claude_code=StubProvider("claude_code"), codex=StubProvider("codex"))
    assert registry.select().name == "claude_code"
    registry = _registry(claude_code=StubProvider("claude_code", signed_in=False), codex=StubProvider("codex"))
    assert registry.select("auto").name == "codex"
    registry = _registry(
        claude_code=StubProvider("claude_code", installed=False, signed_in=False),
        codex=StubProvider("codex", signed_in=False),
    )
    assert isinstance(registry.select(None), ExtractiveProvider)


def test_registry_named_provider_that_is_not_usable_raises():
    registry = _registry(
        claude_code=StubProvider("claude_code", signed_in=False, fix="claude auth login"),
        codex=StubProvider("codex", installed=False, signed_in=False, fix="npm install -g @openai/codex"),
    )
    with pytest.raises(ProviderError) as err:
        registry.select("claude_code")
    assert err.value.kind == "not_signed_in" and err.value.fix_command == "claude auth login"
    with pytest.raises(ProviderError) as err:
        registry.select("codex")
    assert err.value.kind == "not_installed"
    assert registry.select("extractive").name == "extractive"
    with pytest.raises(ValueError):
        registry.select("gemini")


def test_registry_none_uses_the_configured_default():
    registry = _registry(Settings(PROVIDER="extractive"), claude_code=StubProvider("claude_code"))
    assert registry.select(None).name == "extractive"
    assert registry.select("claude_code").name == "claude_code"


def test_registry_statuses_order_and_cache():
    claude, codex = StubProvider("claude_code"), StubProvider("codex")
    registry = _registry(claude_code=claude, codex=codex)
    assert [s.name for s in registry.statuses()] == ["claude_code", "codex", "extractive"]
    registry.statuses()
    registry.select()
    assert claude.status_calls == 1 and codex.status_calls == 1
    registry.statuses(refresh=True)
    assert claude.status_calls == 2


def test_registry_status_check_failure_is_contained():
    class Broken(StubProvider):
        def status(self) -> ProviderStatus:
            raise RuntimeError("boom")

    registry = _registry(claude_code=Broken("claude_code"), codex=StubProvider("codex"))
    claude_status = registry.statuses()[0]
    assert claude_status.installed is False and claude_status.detail == "The status check failed."
    assert registry.select().name == "codex"


def test_registry_doctor_report():
    registry = _registry(
        claude_code=StubProvider("claude_code", signed_in=False, fix="claude auth login"),
        codex=StubProvider("codex"),
    )
    report = registry.doctor_report()
    assert report["configured"] == "auto"
    assert report["default"] == {"name": "codex", "ok": True, "message": None, "fix_command": None}
    names = [p["name"] for p in report["providers"]]
    assert names == ["claude_code", "codex", "extractive"]
    claude = report["providers"][0]
    assert claude["fix_command"] == "claude auth login" and claude["signed_in"] is False
    assert set(claude) >= {"installed", "signed_in", "version", "fix_command", "binary"}


def test_registry_doctor_report_when_the_configured_provider_is_unusable():
    registry = _registry(Settings(PROVIDER="claude_code"),
                         claude_code=StubProvider("claude_code", signed_in=False, fix="claude auth login"))
    default = registry.doctor_report()["default"]
    assert default["ok"] is False and default["fix_command"] == "claude auth login"


def test_registry_with_fake_clis_end_to_end(tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch):
    claude_shim = make_shim(tmp_path / "bin", "claude", "fake_claude.py")
    codex_shim = make_shim(tmp_path / "bin", "codex", "fake_codex.py")
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", "out")
    registry = ProviderRegistry(Settings(CLAUDE_BIN=str(claude_shim), CODEX_BIN=str(codex_shim), PROVIDER="auto"))
    statuses = {s.name: s for s in registry.statuses()}
    assert statuses["claude_code"].installed and not statuses["claude_code"].signed_in
    assert statuses["codex"].signed_in
    assert registry.select().name == "codex"
