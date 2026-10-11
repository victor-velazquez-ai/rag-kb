"""Run a golden set end to end and build the report.

`run_eval` indexes the golden set's corpus into a throwaway data folder, asks every question through
the same retrieval and composition path the app uses, scores the results and returns a report dict.
It prints nothing; `format_report` turns the report into text for the terminal.

The retriever is built the way the app builds it (`Retriever.from_settings`: `CANDIDATES`,
`NEIGHBOUR_EXPANSION`, `MIN_SCORE`, and the reranker `RERANKER` asks for), and every question is
searched with the composer's budget (`MAX_CONTEXT_CHARS`), as an answer in the app is.

- Offline (`live=False`): the deterministic `FakeEmbedder`, no reranker, no vector-score floor and the
  extractive provider. No model, no download, no network, the same result on every machine; it checks
  the pipeline and the lexical side of search.
- Live (`live=True`): the real embedding model from `RAGKB_EMBED_MODEL`, the reranker from
  `RAGKB_RERANKER` and the provider selected by `RAGKB_PROVIDER`, so a question costs at most one call
  on the user's subscription (none when the reranker finds every passage weak and the answer is
  declined before the call).
"""
from __future__ import annotations

import logging
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragkb.answer.compose import compose
from ragkb.config import Settings
from ragkb.evals.golden import GoldenCase, load_golden_set
from ragkb.evals.scorers import (
    UNMEASURED_STATUSES,
    CaseResult,
    case_outcome,
    gate,
    hit_rank,
    missing_keywords,
    score,
)
from ragkb.ingest.pipeline import IngestPipeline
from ragkb.models import Citation, Hit
from ragkb.providers.base import Provider, ProviderError
from ragkb.providers.extractive import ExtractiveProvider
from ragkb.providers.registry import ProviderRegistry
from ragkb.retrieval.base import ranked_only
from ragkb.retrieval.embedder import FakeEmbedder, FastEmbedEmbedder
from ragkb.retrieval.reranker import reranker_for
from ragkb.retrieval.service import RetrievalUnavailable, Retriever
from ragkb.store.lancedb_store import LanceDBStore
from ragkb.store.sqlite_meta import SQLiteMeta

logger = logging.getLogger(__name__)

OFFLINE_DIMS = 384
_MAX_ERROR_CHARS = 200


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 3] + "..."


def _corpus_files(corpus_dir: Path) -> list[Path]:
    files = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(corpus_dir).parts):
            continue
        files.append(path)
    return files


def _select_live_provider(settings: Settings) -> tuple[Provider | None, str | None]:
    """The provider the app would use for these settings, or None and the reason it is not available."""
    requested = None if settings.PROVIDER.strip().lower() in {"", "auto"} else settings.PROVIDER.strip()
    try:
        return ProviderRegistry(settings).select(requested), None
    except ProviderError as exc:
        return None, exc.message or exc.kind


def _close(resource: Any) -> None:
    closer = getattr(resource, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            logger.debug("closing %s failed", type(resource).__name__, exc_info=True)


class _Run:
    """One run's state: the throwaway stores, the retriever, the document names, the results."""

    def __init__(self, settings: Settings, embedder: Any, provider: Any, top_k: int,
                 provider_problem: str | None = None):
        self.settings = settings
        self.provider = provider
        self.provider_problem = provider_problem
        self.top_k = top_k
        self.meta = SQLiteMeta(settings.meta_path)
        self.store = LanceDBStore(settings.lancedb_dir, embedder.dims)
        self.pipeline = IngestPipeline(settings, self.meta, self.store, embedder)
        self.reranker = reranker_for(settings)
        self.retriever = Retriever.from_settings(self.store, embedder, settings, reranker=self.reranker)
        self.doc_names: dict[str, list[str]] = {}

    def close(self) -> None:
        _close(self.store)
        _close(self.meta)

    def ingest(self, collection_id: str, corpus_dir: Path) -> dict[str, Any]:
        files = _corpus_files(corpus_dir)
        failed: list[dict[str, str]] = []
        for path in files:
            try:
                self.pipeline.ingest_file(collection_id, path)
            except Exception as exc:  # noqa: BLE001 - a file that fails is reported, the run goes on
                failed.append({"file": path.relative_to(corpus_dir).as_posix(), "error": _short_error(exc)})
        documents = self.meta.list_documents(collection_id)
        for doc in documents:
            self.doc_names[doc.id] = [doc.title, doc.filename]
            status = getattr(doc.status, "value", doc.status)
            if status == "failed" and not any(f["file"] == doc.filename for f in failed):
                failed.append({"file": doc.filename, "error": doc.error or "failed"})
        ready = sum(1 for d in documents if getattr(d.status, "value", d.status) == "ready")
        return {"files": len(files), "ready": ready, "failed": failed, "chunks": self.store.count(collection_id)}

    def _names_for_hit(self, hit: Hit) -> list[str]:
        return self.doc_names.get(hit.chunk.doc_id, [hit.chunk.title])

    def _names_for_citation(self, citation: Citation) -> list[str]:
        return self.doc_names.get(citation.doc_id, [citation.title])

    def run_case(self, collection_id: str, case: GoldenCase) -> CaseResult:
        result = CaseResult(
            case_id=case.id,
            kind=case.kind,
            question=case.question,
            expect_sources=list(case.expect_sources),
            expect_keywords=list(case.expect_keywords),
        )
        if self.provider is None:
            result.unmeasured_reason = f"no answer provider: {self.provider_problem or 'not available'}"
            return result
        started = time.perf_counter()
        try:
            hits = self.retriever.search(collection_id, case.question, self.top_k,
                                         budget_chars=self.settings.MAX_CONTEXT_CHARS)
        except RetrievalUnavailable:
            result.unmeasured_reason = "search unavailable"
            return result
        except Exception as exc:  # noqa: BLE001 - recorded as unmeasured
            result.unmeasured_reason = f"search failed: {_short_error(exc)}"
            return result
        try:
            answer = compose(
                case.question,
                hits,
                self.provider,
                self.settings,
                collection_id=collection_id,
                knowledge_version=self.meta.current_version(),
                has_documents=self.store.has_documents(collection_id),
            )
        except Exception as exc:  # noqa: BLE001 - recorded as unmeasured
            result.unmeasured_reason = f"answer failed: {_short_error(exc)}"
            return result
        finally:
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = getattr(answer.status, "value", answer.status)
        result.status = status
        if status in UNMEASURED_STATUSES:
            result.unmeasured_reason = answer.message or status
        # Ranks count ranked passages only: a passage attached as context to its neighbour is not a rank.
        result.retrieved = [self._names_for_hit(hit) for hit in ranked_only(hits)]
        result.cited = [self._names_for_citation(c) for c in answer.citations]
        result.answer_text = answer.answer or ""
        result.model = answer.provider.model if answer.provider else None
        return result


def _case_row(result: CaseResult) -> dict[str, Any]:
    return {
        "id": result.case_id,
        "kind": result.kind,
        "question": result.question,
        "status": result.status,
        "outcome": case_outcome(result),
        "hit_rank": hit_rank(result) if result.measured and result.kind == "positive" else None,
        "citations": result.citation_count,
        "top_sources": [names[0] for names in result.retrieved[:3]],
        "cited_sources": [names[0] for names in result.cited],
        "expect_sources": result.expect_sources,
        "keywords_missing": missing_keywords(result) if result.measured and result.expect_keywords else [],
        "unmeasured_reason": result.unmeasured_reason,
        "elapsed_ms": result.elapsed_ms,
    }


def run_eval(golden_path: Path, settings: Settings, *, live: bool) -> dict[str, Any]:
    """Index the golden set's corpus in a temporary data folder, run every case, return the report.

    Report keys: golden_path, mode, collection, corpus_dir, top_k, embedder, reranker, provider, model,
    knowledge_version, corpus, cases, totals, gate (outcome, exit_code, reasons, thresholds), outcome
    (the gate outcome again, PASS, FAIL or SKIPPED), started_at, elapsed_ms. Prints nothing.
    """
    golden = load_golden_set(golden_path)
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    top_k = golden.top_k or settings.TOP_K

    with tempfile.TemporaryDirectory(prefix="ragkb-eval-", ignore_cleanup_errors=True) as tmp:
        run_settings = settings.model_copy(update={"DATA_DIR": Path(tmp)})
        provider_problem: str | None = None
        if live:
            embedder: Any = FastEmbedEmbedder(settings.EMBED_MODEL)
            provider, provider_problem = _select_live_provider(run_settings)
        else:
            # Deterministic: no reranker model (nothing to download) and no floor on the fake cosines.
            run_settings = run_settings.model_copy(update={"RERANKER": "none", "MIN_SCORE": 0.0})
            embedder = FakeEmbedder(dims=OFFLINE_DIMS)
            provider = ExtractiveProvider()
        run_settings.ensure_dirs()
        run = _Run(run_settings, embedder, provider, top_k, provider_problem)
        reranker_name = getattr(run.reranker, "name", None) or "none"
        try:
            collection = run.meta.create_collection(golden.collection, "golden set evaluation")
            corpus = run.ingest(collection.id, golden.corpus_dir)
            results = [run.run_case(collection.id, case) for case in golden.cases]
            knowledge_version = run.meta.current_version()
        finally:
            run.close()

    totals = score(results, top_k)
    verdict = gate(totals)
    models = sorted({r.model for r in results if r.model})
    return {
        "golden_path": str(Path(golden_path)),
        "mode": "live" if live else "offline",
        "collection": golden.collection,
        "corpus_dir": str(golden.corpus_dir),
        "top_k": top_k,
        "embedder": getattr(embedder, "name", type(embedder).__name__),
        "reranker": reranker_name,
        "provider": getattr(provider, "name", None) if provider is not None else None,
        "model": ", ".join(models) if models else None,
        "knowledge_version": knowledge_version,
        "corpus": corpus,
        "cases": [_case_row(r) for r in results],
        "totals": totals,
        "gate": verdict,
        "outcome": verdict["outcome"],
        "started_at": started_at.isoformat(),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------- text report


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def format_report(report: dict[str, Any]) -> str:
    """The report as plain text: header, one row per case, totals and the gate verdict."""
    totals = report["totals"]
    corpus = report["corpus"]
    gate_info = report["gate"]
    lines = [
        f"rag-kb eval  {report['golden_path']}",
        (
            f"mode {report['mode']}: embedder {report['embedder']}, reranker {report.get('reranker') or 'none'}, "
            f"provider {report['provider'] or 'none'}"
            + (f" ({report['model']})" if report.get("model") else "")
        ),
        (
            f"corpus {report['corpus_dir']}: {corpus['files']} files, {corpus['ready']} ready, "
            f"{len(corpus['failed'])} failed, {corpus['chunks']} passages; "
            f"knowledge version {report['knowledge_version']}; top_k {report['top_k']}"
        ),
    ]
    for failure in corpus["failed"]:
        lines.append(f"  failed to index {failure['file']}: {failure['error']}")
    lines.append("")

    header = f"{'id':<8} {'kind':<4} {'outcome':<12} {'status':<22} {'rank':>4} {'cites':>5} {'ms':>7}  question"
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["cases"]:
        rank = "" if row["hit_rank"] is None else str(row["hit_rank"])
        status = row["status"] or "-"
        lines.append(
            f"{_clip(row['id'], 8):<8} {row['kind'][:3]:<4} {row['outcome']:<12} {status:<22} {rank:>4} "
            f"{row['citations']:>5} {row['elapsed_ms']:>7}  {_clip(row['question'], 60)}"
        )
        if row["outcome"] == "miss":
            found = ", ".join(row["top_sources"]) or "nothing"
            lines.append(f"{'':<8} expected {', '.join(row['expect_sources'])}; top results: {found}")
        if row["keywords_missing"]:
            lines.append(f"{'':<8} answer lacks: {', '.join(row['keywords_missing'])}")
        if row["unmeasured_reason"]:
            lines.append(f"{'':<8} not measured: {row['unmeasured_reason']}")
    lines.append("")

    recall_keys = sorted((k for k in totals if k.startswith("recall_at_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    lines.append(
        "  ".join(f"recall@{k.rsplit('_', 1)[1]} {_num(totals[k])}" for k in recall_keys)
        + f"  MRR {_num(totals['mrr'])}"
    )
    lines.append(
        f"answer rate {_num(totals['answer_rate'])}  citation rate {_num(totals['citation_rate'])}  "
        f"keyword rate {_num(totals['keyword_rate'])}"
    )
    lines.append(
        f"false-answer rate {_num(totals['false_answer_rate'])}  decline rate {_num(totals['decline_rate'])}"
    )
    lines.append(
        f"ran {totals['ran']} of {totals['cases']} cases ({totals['positives_ran']} of {totals['positives']} "
        f"positives, {totals['negatives_ran']} of {totals['negatives']} negatives); unmeasured {totals['unmeasured']}"
    )
    lines.append(f"elapsed {report['elapsed_ms'] / 1000:.1f} s")
    lines.append("")
    lines.append(f"GATE {gate_info['outcome']}: " + "; ".join(gate_info["reasons"]))
    return "\n".join(lines)
