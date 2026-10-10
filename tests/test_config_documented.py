"""Every setting is documented in .env.example, and .env.example documents nothing that does not exist."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from ragkb.config import Settings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
PACKAGE = ROOT / "ragkb"
PREFIX = str(Settings.model_config["env_prefix"])

_KEY_LINE = re.compile(r"^(RAGKB_[A-Z0-9_]+)=", re.MULTILINE)
# A quoted variable name in a module that reads the environment: os.environ.get("RAGKB_X"),
# os.getenv("RAGKB_X"), or a constant holding the name for such a read.
_QUOTED_NAME = re.compile(r"[\"'](RAGKB_[A-Z0-9_]+)[\"']")


def documented_keys() -> list[str]:
    return _KEY_LINE.findall(ENV_EXAMPLE.read_text(encoding="utf-8"))


def settings_keys() -> set[str]:
    return {f"{PREFIX}{name}" for name in Settings.model_fields}


def names_read_from_the_environment() -> set[str]:
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "os.environ" in text or "getenv" in text:
            names |= set(_QUOTED_NAME.findall(text))
    return names


def test_the_scan_finds_the_known_environment_reads() -> None:
    assert PREFIX == "RAGKB_"
    assert "RAGKB_FAKE_EMBEDDER" in names_read_from_the_environment()


def test_every_setting_is_documented() -> None:
    missing = sorted(settings_keys() - set(documented_keys()))
    assert not missing, f"add these to .env.example: {missing}"


def test_every_documented_key_is_a_setting_or_read_by_the_code() -> None:
    unknown = sorted(set(documented_keys()) - settings_keys() - names_read_from_the_environment())
    assert not unknown, f".env.example documents keys nothing reads: {unknown}"


def test_every_variable_the_code_reads_is_documented() -> None:
    missing = sorted(names_read_from_the_environment() - set(documented_keys()))
    assert not missing, f"add these to .env.example: {missing}"


def test_no_key_is_documented_twice() -> None:
    keys = documented_keys()
    assert len(keys) == len(set(keys))


def test_documented_values_are_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(PREFIX):
            monkeypatch.delenv(name)
    assert Settings(_env_file=ENV_EXAMPLE) == Settings(_env_file=None)
