"""A fake `codex` binary that replays recorded Codex outputs.

Scenarios (FAKE_CODEX_SCENARIO): success (default), not_logged_in, rate_limited, hang, echo_error.
FAKE_CODEX_AUTH=out makes `login status` report a signed-out CLI. FAKE_CODEX_RESULT replaces the answer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

RATE_LIMITED = (
    "Reading prompt from stdin...\n"
    "ERROR: You've hit your usage limit. Upgrade to Pro or try again in 2 hours 13 minutes.\n"
)


def option(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


def record(args: list[str], stdin: str) -> None:
    path = os.environ.get("FAKE_CLI_RECORD")
    if not path:
        return
    workdir = option(args, "-C")
    data = {
        "argv": args,
        "stdin": stdin,
        "cwd": os.getcwd(),
        "workdir_exists": bool(workdir and Path(workdir).is_dir()),
        "env_openai_api_key": "OPENAI_API_KEY" in os.environ,
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "success")
    if args[:1] in (["--version"], ["-V"]):
        print("codex-cli 0.162.0-alpha.2")
        return 0
    if args[:2] == ["login", "status"]:
        if os.environ.get("FAKE_CODEX_AUTH", "in") == "out":
            print("Not logged in", file=sys.stderr)
            return 1
        print("Logged in using ChatGPT", file=sys.stderr)
        return 0
    if args[:1] != ["exec"]:
        print("error: only exec is faked", file=sys.stderr)
        return 2

    stdin = sys.stdin.buffer.read().decode("utf-8")
    record(args, stdin)
    if scenario == "hang":
        time.sleep(60)
        return 0
    if scenario == "not_logged_in":
        sys.stderr.write((FIXTURES / "codex_not_logged_in.stderr.txt").read_text(encoding="utf-8"))
        return 1
    if scenario == "rate_limited":
        sys.stderr.write(RATE_LIMITED)
        return 1
    if scenario == "echo_error":
        # Codex echoes the prompt; an error after it must be judged on the error, not on the passages.
        sys.stderr.write("Reading prompt from stdin...\nuser\n" + stdin + "\nERROR: stream disconnected before completion\n")
        return 1
    answer = os.environ.get("FAKE_CODEX_RESULT")
    if answer is None:
        answer = (FIXTURES / "codex_success.answer.md").read_text(encoding="utf-8")
    outfile = option(args, "-o")
    if outfile:
        Path(outfile).write_text(answer, encoding="utf-8")
    sys.stderr.write((FIXTURES / "codex_success.stderr.txt").read_text(encoding="utf-8"))
    sys.stdout.write(answer + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
