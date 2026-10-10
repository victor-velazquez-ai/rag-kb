"""A fake `claude` binary that replays recorded Claude Code outputs.

Scenarios (FAKE_CLAUDE_SCENARIO): success (default), not_logged_in, rate_limited, hang, crash,
haiku_first, old_help. FAKE_CLAUDE_AUTH=out makes `auth status` report a signed-out CLI.
FAKE_CLAUDE_RESULT replaces the answer text of the success scenario.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

HELP_FULL = """Usage: claude [options] [command] [prompt]

Options:
  -p, --print                   Print response and exit
  --model <model>               Model for the current session
  --output-format <format>      Output format
  --no-session-persistence      Disable session persistence
  --system-prompt <prompt>      System prompt to use for the session
  --safe-mode                   Start with all customizations disabled
  --strict-mcp-config           Only use MCP servers from --mcp-config
  --tools <tools...>            Use "" to disable all tools
"""

HELP_OLD = """Usage: claude [options] [command] [prompt]

Options:
  -p, --print                   Print response and exit
  --model <model>               Model for the current session
  --output-format <format>      Output format
  --system-prompt <prompt>      System prompt to use for the session
"""


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def record(args: list[str], stdin: str) -> None:
    path = os.environ.get("FAKE_CLI_RECORD")
    if not path:
        return
    system_prompt = None
    if "--system-prompt-file" in args:
        target = Path(args[args.index("--system-prompt-file") + 1])
        system_prompt = target.read_text(encoding="utf-8") if target.is_file() else None
    data = {
        "argv": args,
        "stdin": stdin,
        "system_prompt": system_prompt,
        "cwd": os.getcwd(),
        "env_anthropic_api_key": "ANTHROPIC_API_KEY" in os.environ,
        "env_claudecode": "CLAUDECODE" in os.environ,
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "success")
    if args[:1] in (["--version"], ["-v"]):
        print("2.1.292 (Claude Code)")
        return 0
    if args[:1] in (["--help"], ["-h"]):
        print(HELP_OLD if scenario == "old_help" else HELP_FULL)
        return 0
    if args[:2] == ["auth", "status"]:
        if os.environ.get("FAKE_CLAUDE_AUTH", "in") == "out":
            emit(fixture("claude_auth_status_logged_out.json"))
            return 1
        emit(fixture("claude_auth_status_logged_in.json"))
        return 0
    if "-p" not in args:
        print("error: interactive mode is not faked", file=sys.stderr)
        return 2

    stdin = sys.stdin.buffer.read().decode("utf-8")
    record(args, stdin)
    if scenario == "hang":
        time.sleep(60)
        return 0
    if scenario == "crash":
        print("Error: something broke while starting", file=sys.stderr)
        return 2
    if scenario == "not_logged_in":
        emit(fixture("claude_not_logged_in.json"))
        return 1
    if scenario == "rate_limited":
        emit(fixture("claude_rate_limited_synthetic.json"))
        return 1
    payload = fixture("claude_success.json")
    if os.environ.get("FAKE_CLAUDE_RESULT") is not None:
        payload["result"] = os.environ["FAKE_CLAUDE_RESULT"]
    if scenario == "haiku_first":
        payload["modelUsage"] = {"claude-haiku-4-5": {"outputTokens": 3}, **payload["modelUsage"]}
    emit(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
