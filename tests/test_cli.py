"""Command line: argument parsing, exit codes, doctor smoke, and ingest plus ask with the fake embedder."""
from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from ragkb import cli

CORE_PARTS = (
    "ragkb.store.sqlite_meta",
    "ragkb.store.lancedb_store",
    "ragkb.retrieval.embedder",
    "ragkb.retrieval.service",
    "ragkb.ingest.parsers",
    "ragkb.ingest.pipeline",
)
ANSWER_PARTS = ("ragkb.providers.registry", "ragkb.answer.compose")


def _require(*modules: str) -> None:
    for name in modules:
        pytest.importorskip(name)


def _all_parts_present() -> bool:
    try:
        for name in (*CORE_PARTS, *ANSWER_PARTS):
            importlib.import_module(name)
    except Exception:  # noqa: BLE001 - any import failure means the part is absent
        return False
    return True


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RAGKB_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("RAGKB_PROVIDER", "extractive")
    monkeypatch.setenv("RAGKB_LOG_LEVEL", "WARNING")


def test_parse_serve():
    args = cli.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9000", "--open", "--reload"])
    assert (args.command, args.host, args.port, args.open, args.reload) == ("serve", "0.0.0.0", 9000, True, True)


def test_parse_ingest_and_ask():
    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "a.md", "docs", "--collection", "Team", "--data-dir", "d"])
    assert args.paths == ["a.md", "docs"] and args.collection == "Team" and args.data_dir == "d"
    assert parser.parse_args(["ingest", "x"]).collection == cli.DEFAULT_COLLECTION

    args = parser.parse_args(["ask", "What is it?", "--provider", "codex", "--model", "opus", "--top-k", "3"])
    assert (args.question, args.provider, args.model, args.top_k) == ("What is it?", "codex", "opus", 3)
    assert parser.parse_args(["ask", "q"]).provider == "auto"


def test_parse_doctor_and_eval():
    parser = cli.build_parser()
    assert parser.parse_args(["doctor", "--json"]).json is True
    args = parser.parse_args(["eval", "--golden", "g.yaml", "--live"])
    assert (args.golden, args.live) == ("g.yaml", True)


def test_usage_errors_exit_2(capsys):
    assert cli.main([]) == 2
    assert cli.main(["frobnicate"]) == 2
    assert cli.main(["ask"]) == 2
    assert cli.main(["ask", "q", "--provider", "gpt"]) == 2
    assert cli.main(["--version"]) == 0
    capsys.readouterr()


def test_ingest_missing_path_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["ingest", str(tmp_path / "missing"), "--data-dir", str(tmp_path / "data")]) == 2
    assert "Not found" in capsys.readouterr().err


def test_serve_runs_uvicorn_with_the_factory(tmp_path, monkeypatch, capsys):
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.delenv("RAGKB_DATA_DIR", raising=False)
    assert cli.main(["serve", "--port", "9123", "--data-dir", str(tmp_path / "data")]) == 0
    (args, kwargs), = calls
    assert args == ("ragkb.server.app:create_app",)
    assert kwargs["factory"] is True and kwargs["port"] == 9123 and kwargs["host"] == "127.0.0.1"
    assert "9123" in capsys.readouterr().out
    assert cli.main(["serve", "--port", "70000"]) == 2


def test_doctor_smoke(tmp_path, capsys):
    rc = cli.main(["doctor", "--data-dir", str(tmp_path / "data"), "--json"])
    report = json.loads(capsys.readouterr().out)
    names = {c["name"] for c in report["checks"]}
    assert {"Python", "Data folder", "Metadata store", "Vector store", "Embedding model"} <= names
    assert rc in (0, 1)
    assert report["ok"] is (rc == 0)
    if _all_parts_present():
        assert rc == 0, report
        assert any(p["name"] == "extractive" for p in report["providers"])


def test_doctor_text_output(tmp_path, capsys):
    cli.main(["doctor", "--data-dir", str(tmp_path / "data")])
    out = capsys.readouterr().out
    assert "Python" in out and "Data folder" in out


def test_eval_needs_a_golden_set(tmp_path, capsys):
    assert cli.main(["eval", "--golden", str(tmp_path / "missing.yaml")]) == 2
    capsys.readouterr()


def test_eval_prints_the_report_and_exits_with_the_gate_code(tmp_path, monkeypatch, capsys):
    golden = tmp_path / "golden.yaml"
    golden.write_text("cases: []\n", encoding="utf-8")
    seen: dict[str, Any] = {}
    outcome = {"value": "PASS", "code": 0}

    def fake_run(path, settings, *, live):
        seen.update(path=path, live=live)
        return {"gate": {"outcome": outcome["value"], "exit_code": outcome["code"], "reasons": []}}

    monkeypatch.setattr(cli, "run_eval", fake_run)
    monkeypatch.setattr(cli, "format_report", lambda report: f"GATE {report['gate']['outcome']}: formatted")
    assert cli.main(["eval", "--golden", str(golden), "--live"]) == 0
    assert seen == {"path": golden, "live": True}
    assert capsys.readouterr().out.strip() == "GATE PASS: formatted"

    for value, code in (("FAIL", 1), ("SKIPPED", 2)):
        outcome.update(value=value, code=code)
        assert cli.main(["eval", "--golden", str(golden)]) == code
        assert capsys.readouterr().out.strip() == f"GATE {value}: formatted"

    assert cli.main(["eval", "--golden", str(golden), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["gate"]["outcome"] == "SKIPPED"


def test_eval_offline_on_the_sample_golden_set_passes(tmp_path, capsys):
    _require(*CORE_PARTS, *ANSWER_PARTS)
    golden = cli._default_golden()
    if golden is None:
        pytest.skip("the sample golden set is not present")
    assert cli.main(["eval", "--golden", str(golden), "--data-dir", str(tmp_path / "data")]) == 0
    out = capsys.readouterr().out
    assert "GATE PASS" in out and "false-answer rate 0.00" in out


def test_ingest_then_ask(tmp_path, capsys):
    _require(*CORE_PARTS, *ANSWER_PARTS)
    docs = tmp_path / "docs"
    (docs / "nested").mkdir(parents=True)
    (docs / "leave.md").write_text(
        "# Leave policy\n\nEvery employee gets 25 days of annual leave per calendar year.\n", encoding="utf-8"
    )
    (docs / "nested" / "expenses.txt").write_text(
        "Expense claims must be filed within 30 days of the purchase.\n", encoding="utf-8"
    )
    (docs / "nested" / "ignore.bin").write_bytes(b"\x00\x01")
    data = str(tmp_path / "data")

    assert cli.main(["ingest", str(docs), "--collection", "Handbook", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "leave.md" in out and "nested/expenses.txt" in out and "ignore.bin" not in out
    assert "2 indexed" in out

    assert cli.main(["ingest", str(docs / "leave.md"), "--collection", "Handbook", "--data-dir", data]) == 0
    assert "already indexed" in capsys.readouterr().out

    rc = cli.main(["ask", "How many days of annual leave?", "--data-dir", data, "--provider", "extractive"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "passages considered" in out

    rc = cli.main(["ask", "How many days of annual leave?", "--data-dir", data, "--json"])
    body = json.loads(capsys.readouterr().out)
    assert rc == 0 and body["status"] in ("answered", "no_answer")

    assert cli.main(["ask", "Anything?", "--collection", "Nope", "--data-dir", data]) == 1
    assert "Collection not found" in capsys.readouterr().err
