"""Command line: argument parsing, exit codes, doctor smoke, and ingest plus ask with the fake embedder."""
from __future__ import annotations

import importlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from ragkb import cli
from ragkb.models import Answer, AnswerStatus, Citation
from ragkb.server.deps import Deps
from tests.test_api import PART_A, PART_A_REASON, FakePipeline, MemoryMeta

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
    # The reranker is on by default; the offline tests never download its model.
    monkeypatch.setenv("RAGKB_RERANKER", "none")


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
    captured = capsys.readouterr()
    assert "Python" in captured.out and "Data folder" in captured.out
    assert "Reranker: off (RAGKB_RERANKER=none)" in captured.out
    assert "embedding model, about 130 MB)" in captured.err and "reranker" not in captured.err


def test_doctor_loads_the_reranker_once_and_names_the_download(tmp_path, monkeypatch, capsys):
    reranker_mod = pytest.importorskip("ragkb.retrieval.reranker")
    loaded: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            loaded.append(model_name)

    monkeypatch.setenv("RAGKB_RERANKER", "local")
    monkeypatch.setattr(reranker_mod, "TextCrossEncoder", FakeCrossEncoder)
    cli.main(["doctor", "--data-dir", str(tmp_path / "data")])
    captured = capsys.readouterr()
    assert loaded == ["Xenova/ms-marco-MiniLM-L-6-v2"]
    assert "and the reranker, about 90 MB) ..." in captured.err
    line = next(line for line in captured.out.splitlines() if "Reranker:" in line)
    assert line.startswith("[  ok] Reranker: local (Xenova/ms-marco-MiniLM-L-6-v2), loaded in ")
    assert "about 90 MB, downloaded on the first run" in line

    def offline_hub(model_name: str) -> None:
        raise OSError("no connection to the model hub")

    monkeypatch.setattr(reranker_mod, "TextCrossEncoder", offline_hub)
    assert cli.main(["doctor", "--data-dir", str(tmp_path / "data2"), "--json"]) in (0, 1)
    check = next(c for c in json.loads(capsys.readouterr().out)["checks"] if c["name"] == "Reranker")
    assert check["ok"] is False and check["essential"] is False
    assert "search keeps the fused order" in check["detail"]


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


def test_ingest_reports_each_outcome_and_a_summary(tmp_path, monkeypatch, capsys):
    meta = MemoryMeta()
    pipeline = FakePipeline(meta)
    monkeypatch.setattr(cli, "build_deps", lambda settings, **kwargs: Deps(settings=settings, meta=meta,
                                                                             pipeline=pipeline))
    docs = tmp_path / "handbook"
    docs.mkdir()
    (docs / "leave.md").write_text("# Leave\n\nVersion one.\n", encoding="utf-8")
    (docs / "expenses.txt").write_text("Expenses are paid monthly.\n", encoding="utf-8")
    data = str(tmp_path / "data")

    assert cli.main(["ingest", str(docs), "--collection", "Team", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "2 ready, 0 replaced, 0 duplicates (already indexed), 0 failed, 0 skipped" in out
    assert sorted(c["source_path"] for c in pipeline.calls) == ["handbook/expenses.txt", "handbook/leave.md"]

    (docs / "leave.md").write_text("# Leave\n\nVersion two.\n", encoding="utf-8")
    (docs / "broken-scan.md").write_text("unreadable", encoding="utf-8")
    assert cli.main(["ingest", str(docs), "--collection", "Team", "--data-dir", data]) == 1
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("handbook/")}
    assert "ready" in rows["handbook/leave.md"] and "replaced leave" in rows["handbook/leave.md"]
    assert "already indexed" in rows["handbook/expenses.txt"]
    assert "failed" in rows["handbook/broken-scan.md"] and "could not be read" in rows["handbook/broken-scan.md"]
    assert "0 ready, 1 replaced, 1 duplicates (already indexed), 1 failed, 0 skipped" in out


def test_ask_prints_the_closest_passages_when_not_covered():
    now = datetime.now(UTC)
    closest = [Citation(marker=1, chunk_id="d1#0", doc_id="d1", title="Office rules", page=2, section="Pets",
                        snippet="Dogs are not allowed in the office, except assistance dogs.", score=0.42)]
    declined = Answer(id=uuid.uuid4().hex, question="Can I bring my cat?", collection_id="c1",
                      status=AnswerStatus.no_answer, message="The knowledge base does not cover this question.",
                      closest=closest, created_at=now)
    text = cli.format_answer(declined)
    assert text.startswith("Not covered: The knowledge base does not cover this question.")
    assert "Closest passages" in text and "1. Office rules (p. 2, Pets)" in text
    assert "Dogs are not allowed in the office" in text

    answered = declined.model_copy(update={"status": AnswerStatus.answered, "answer": "No pets [1].",
                                           "message": None})
    assert "Closest passages" not in cli.format_answer(answered)


def test_ingest_then_ask(tmp_path, capsys):
    _require(*CORE_PARTS, *ANSWER_PARTS)
    if not PART_A:
        pytest.skip(PART_A_REASON)
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
    assert "docs/leave.md" in out and "docs/nested/expenses.txt" in out and "ignore.bin" not in out
    assert "2 ready, 0 replaced, 0 duplicates" in out

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
