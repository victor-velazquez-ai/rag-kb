"""Evaluation harness: the golden-set loader, the scorers and gate, and an offline run over the samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragkb.config import Settings
from ragkb.evals.golden import GoldenSetError, load_golden_set
from ragkb.evals.scorers import (
    CaseResult,
    case_outcome,
    gate,
    hit_rank,
    mean_reciprocal_rank,
    recall_at,
    score,
    source_matches,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_GOLDEN = REPO / "samples" / "golden.yaml"
SAMPLE_CORPUS = REPO / "samples" / "corpus"

VALID = """\
corpus_dir: corpus
collection: demo
cases:
  - id: q01
    question: "How many days of leave?"
    expect_sources: ["handbook.md"]
    expect_keywords: ["27"]
  - id: n01
    question: "What is the capital of Mars?"
    expect: empty
"""


def write_set(tmp_path: Path, text: str, *, corpus: bool = True) -> Path:
    if corpus:
        (tmp_path / "corpus").mkdir(exist_ok=True)
    path = tmp_path / "golden.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- loader


def test_valid_set_loads(tmp_path: Path) -> None:
    golden = load_golden_set(write_set(tmp_path, VALID))
    assert golden.collection == "demo"
    assert golden.corpus_dir == (tmp_path / "corpus").resolve()
    assert [c.id for c in golden.positives] == ["q01"]
    assert [c.id for c in golden.negatives] == ["n01"]
    assert golden.cases[0].expect_keywords == ["27"]
    assert golden.cases[1].kind == "negative"


def test_corpus_dir_is_found_from_a_parent_folder(tmp_path: Path) -> None:
    (tmp_path / "samples" / "corpus").mkdir(parents=True)
    path = tmp_path / "samples" / "golden.yaml"
    path.write_text(VALID.replace("corpus_dir: corpus", "corpus_dir: samples/corpus"), encoding="utf-8")
    assert load_golden_set(path).corpus_dir == (tmp_path / "samples" / "corpus").resolve()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda t: t.replace("collection: demo", "collection: demo\nmodel: big"), "unknown field"),
        (lambda t: t.replace('    expect_keywords: ["27"]', '    expect_keyword: ["27"]'), "case 'q01'"),
        (lambda t: t.replace('    expect_sources: ["handbook.md"]\n', ""), "at least one entry in expect_sources"),
        (lambda t: t.replace("    expect: empty", "    expect: empty\n    expect_sources: [a.md]"), "must not list"),
        (lambda t: t.replace("id: n01", "id: q01"), "repeated: q01"),
        (lambda t: t.replace("    expect: empty", "    expect: nothing"), "case 'n01'"),
        (lambda t: t.replace('question: "What is the capital of Mars?"', 'question: "   "'), "blank"),
        (lambda t: t.replace("corpus_dir: corpus", "corpus_dir: nowhere"), "was not found"),
        (lambda t: "cases: [", "not valid YAML"),
        (lambda t: "- just a list", "must be a mapping"),
        (lambda t: "corpus_dir: corpus\ncases: []", "cases"),
    ],
)
def test_bad_sets_fail_loudly(tmp_path: Path, change: Any, expected: str) -> None:
    with pytest.raises(GoldenSetError) as info:
        load_golden_set(write_set(tmp_path, change(VALID)))
    assert expected in str(info.value)


def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="cannot read"):
        load_golden_set(tmp_path / "absent.yaml")


def test_sample_golden_set_loads_and_names_real_documents() -> None:
    golden = load_golden_set(SAMPLE_GOLDEN)
    assert golden.corpus_dir == SAMPLE_CORPUS.resolve()
    assert len(golden.positives) >= 10
    assert len(golden.negatives) >= 3
    files = [p.name for p in SAMPLE_CORPUS.iterdir() if p.is_file()] if SAMPLE_CORPUS.is_dir() else []
    if not files:
        pytest.skip("the sample corpus is not there yet")
    for case in golden.positives:
        assert any(source_matches([src], [name]) for src in case.expect_sources for name in files), case.id


# ---------------------------------------------------------------- scorers


def pos(case_id: str, retrieved: list[str], *, cited: list[str] | None = None, status: str = "answered",
        expect: str = "handbook.md", keywords: list[str] | None = None, text: str = "",
        unmeasured: str | None = None) -> CaseResult:
    return CaseResult(
        case_id=case_id, kind="positive", expect_sources=[expect], expect_keywords=keywords or [],
        status=status, unmeasured_reason=unmeasured,
        retrieved=[[name] for name in retrieved], cited=[[name] for name in (cited or [])], answer_text=text,
    )


def neg(case_id: str, status: str, *, unmeasured: str | None = None) -> CaseResult:
    return CaseResult(case_id=case_id, kind="negative", status=status, unmeasured_reason=unmeasured)


def test_source_matching_ignores_case_folders_and_extension() -> None:
    assert source_matches(["Employee-Handbook.md"], ["employee-handbook.md"])
    assert source_matches(["employee-handbook"], ["Employee Handbook", "employee-handbook.md"])
    assert source_matches(["Employee Handbook"], ["Employee Handbook"])
    assert source_matches(["policies/travel.md"], ["travel.md"])
    assert not source_matches(["travel.md"], ["travel-faq.md"])
    assert not source_matches([""], [""])


def test_hit_rank_uses_the_better_of_hits_and_citations() -> None:
    assert hit_rank(pos("a", ["x.md", "handbook.md"])) == 2
    assert hit_rank(pos("b", ["x.md", "y.md", "z.md"], cited=["handbook.md"])) == 1
    assert hit_rank(pos("c", ["x.md"], cited=["y.md"])) is None


def hand_made() -> list[CaseResult]:
    return [
        pos("p1", ["handbook.md", "x.md"], cited=["handbook.md"], keywords=["27"], text="You get 27 days [1]."),
        pos("p2", ["a.md", "b.md", "c.md", "handbook.md"], cited=["a.md"], keywords=["leave"], text="Nope [1]."),
        pos("p3", ["a.md", "b.md"], status="no_answer"),
        pos("p4", ["a.md"], cited=[], text="An answer with no citation."),
        pos("p5", [], status="provider_unavailable", unmeasured="Claude Code is not signed in."),
        neg("n1", "no_answer"),
        neg("n2", "answered"),
        neg("n3", "no_documents"),
        neg("n4", "rate_limited", unmeasured="usage limit"),
    ]


def test_scores_on_hand_made_results() -> None:
    results = hand_made()
    assert recall_at(results, 1) == 0.25
    assert recall_at(results, 3) == 0.25
    assert recall_at(results, 5) == 0.5
    assert mean_reciprocal_rank(results) == round((1 + 1 / 4) / 4, 4)

    totals = score(results, top_k=6)
    assert totals["cases"] == 9
    assert (totals["positives"], totals["negatives"]) == (5, 4)
    assert (totals["positives_ran"], totals["negatives_ran"]) == (4, 3)
    assert (totals["ran"], totals["unmeasured"]) == (7, 2)
    assert totals["recall_at_6"] == 0.5
    assert totals["answer_rate"] == 0.75
    assert totals["citation_rate"] == round(2 / 3, 4)
    assert totals["keyword_rate"] == 0.5
    assert totals["false_answer_rate"] == round(1 / 3, 4)
    assert totals["decline_rate"] == round(2 / 3, 4)


def test_case_outcomes() -> None:
    by_id = {r.case_id: case_outcome(r) for r in hand_made()}
    assert by_id == {
        "p1": "ok", "p2": "ok", "p3": "miss", "p4": "miss", "p5": "unmeasured",
        "n1": "ok", "n2": "false_answer", "n3": "ok", "n4": "unmeasured",
    }
    assert case_outcome(pos("p6", ["handbook.md"], status="no_answer")) == "declined"


def test_status_alone_marks_a_case_unmeasured() -> None:
    result = pos("p", ["handbook.md"], status="retrieval_unavailable")
    assert result.measured is False
    assert score([result])["recall_at_5"] is None


def test_gate_pass_fail_and_skipped() -> None:
    good = [pos(f"p{i}", ["handbook.md"], cited=["handbook.md"]) for i in range(9)] + [
        pos("p9", ["x.md"]), neg("n1", "no_answer"), neg("n2", "no_answer")]
    verdict = gate(score(good))
    assert (verdict["outcome"], verdict["exit_code"]) == ("PASS", 0)

    low_recall = [pos(f"p{i}", ["x.md"]) for i in range(3)] + [pos("p3", ["handbook.md"]), neg("n1", "no_answer")]
    verdict = gate(score(low_recall))
    assert (verdict["outcome"], verdict["exit_code"]) == ("FAIL", 1)
    assert any("recall@5" in reason for reason in verdict["reasons"])

    false_answers = [pos("p0", ["handbook.md"]), neg("n1", "answered"), neg("n2", "no_answer")]
    verdict = gate(score(false_answers))
    assert verdict["outcome"] == "FAIL"
    assert any("false-answer" in reason for reason in verdict["reasons"])


def test_gate_never_passes_on_too_little() -> None:
    mostly_unmeasured = [pos("p0", ["handbook.md"]), neg("n0", "no_answer")] + [
        pos(f"u{i}", [], status="provider_unavailable", unmeasured="not signed in") for i in range(3)]
    verdict = gate(score(mostly_unmeasured))
    assert (verdict["outcome"], verdict["exit_code"]) == ("SKIPPED", 2)

    no_negatives = [pos("p0", ["handbook.md"]), pos("p1", ["handbook.md"])]
    verdict = gate(score(no_negatives))
    assert verdict["outcome"] == "SKIPPED"
    assert any("false-answer" in reason for reason in verdict["reasons"])

    assert gate(score([]))["outcome"] == "SKIPPED"


# ---------------------------------------------------------------- report and offline run


def test_format_report_on_a_synthetic_report() -> None:
    run_mod = pytest.importorskip("ragkb.evals.run")
    results = hand_made()
    totals = score(results, top_k=6)
    report = {
        "golden_path": "golden.yaml", "mode": "offline", "collection": "demo", "corpus_dir": "corpus",
        "top_k": 6, "embedder": "fake", "provider": "extractive", "model": None, "knowledge_version": "v1",
        "corpus": {"files": 3, "ready": 2, "failed": [{"file": "bad.pdf", "error": "empty"}], "chunks": 9},
        "cases": [run_mod._case_row(r) for r in results], "totals": totals, "gate": gate(totals),
        "started_at": "now", "elapsed_ms": 1234,
    }
    text = run_mod.format_report(report)
    assert "GATE FAIL" in text
    assert "false_answer" in text and "unmeasured" in text
    assert "failed to index bad.pdf" in text
    assert "recall@5 0.50" in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_offline_run_over_the_sample_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    for module in ("ragkb.store.sqlite_meta", "ragkb.store.lancedb_store", "ragkb.retrieval.embedder",
                   "ragkb.retrieval.service", "ragkb.ingest.pipeline", "ragkb.answer.compose",
                   "ragkb.providers.extractive", "ragkb.providers.registry"):
        pytest.importorskip(module)
    run_mod = pytest.importorskip("ragkb.evals.run")
    if not SAMPLE_CORPUS.is_dir() or not any(SAMPLE_CORPUS.iterdir()):
        pytest.skip("the sample corpus is not there yet")

    report = run_mod.run_eval(SAMPLE_GOLDEN, Settings(_env_file=None), live=False)

    assert capsys.readouterr().out == ""
    for key in ("golden_path", "mode", "collection", "corpus_dir", "top_k", "embedder", "provider", "model",
                "knowledge_version", "corpus", "cases", "totals", "gate", "started_at", "elapsed_ms"):
        assert key in report
    assert report["mode"] == "offline"
    assert report["provider"] == "extractive"
    assert report["corpus"]["failed"] == []
    assert report["corpus"]["ready"] == report["corpus"]["files"] > 0
    golden = load_golden_set(SAMPLE_GOLDEN)
    assert [row["id"] for row in report["cases"]] == [c.id for c in golden.cases]
    totals = report["totals"]
    assert totals["unmeasured"] == 0
    for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "answer_rate", "citation_rate",
                "keyword_rate", "false_answer_rate", "decline_rate"):
        assert key in totals
        assert totals[key] is None or 0.0 <= totals[key] <= 1.0
    assert report["gate"]["outcome"] == "PASS", report["gate"]["reasons"]
    # Every positive answered from its source, every negative declined (no false answers).
    not_ok = [(row["id"], row["outcome"]) for row in report["cases"] if row["outcome"] != "ok"]
    assert not not_ok, not_ok
    assert "GATE PASS" in run_mod.format_report(report)
