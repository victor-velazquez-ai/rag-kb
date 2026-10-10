"""Fake Claude Code and Codex CLIs for the adapter tests.

Each fake is a Python script that replays recorded outputs (see fixtures/). `make_shim` wraps a script
in a launcher the adapters can run like the real binary: a .cmd file on Windows (the same kind of
launcher an npm install creates) or an executable shell script elsewhere. The scenario is picked with
FAKE_CLAUDE_SCENARIO / FAKE_CODEX_SCENARIO, and every model call is recorded as JSON in the file named
by FAKE_CLI_RECORD.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

FAKE_DIR = Path(__file__).resolve().parent
FIXTURES = FAKE_DIR / "fixtures"


def make_shim(directory: Path, name: str, script: str) -> Path:
    """Create an executable launcher called *name* in *directory* that runs fake *script*."""
    directory.mkdir(parents=True, exist_ok=True)
    target = FAKE_DIR / script
    if os.name == "nt":
        shim = directory / f"{name}.cmd"
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{target}" %*\r\n', encoding="utf-8")
    else:
        shim = directory / name
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{target}" "$@"\n', encoding="utf-8")
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim
