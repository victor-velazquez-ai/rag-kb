"""The ``ragkb`` command: serve, ingest, ask, doctor, eval.

Exit codes: 0 ok, 1 failure, 2 usage. `eval` returns its gate's code: 0 PASS, 1 FAIL, 2 SKIPPED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn

from ragkb import __version__
from ragkb.config import Settings
from ragkb.models import Answer, AnswerStatus, DocumentStatus
from ragkb.providers.base import ProviderError
from ragkb.server.app import configure_logging
from ragkb.server.deps import (
    Deps,
    build_deps,
    fake_embedder_requested,
    missing_parts,
    provider_label,
    supported_media_type,
    supported_types,
)
from ragkb.server.errors import ApiError, plain_error
from ragkb.server.loopback import host_is_loopback

try:
    from ragkb.evals.run import format_report, run_eval
except Exception:  # noqa: BLE001 - the evaluation part is optional
    format_report = None  # type: ignore[assignment]
    run_eval = None  # type: ignore[assignment]

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
DEFAULT_COLLECTION = "default"

_STATE_LABELS: dict[str, str] = {
    AnswerStatus.no_answer.value: "Not covered",
    AnswerStatus.no_documents.value: "No documents",
    AnswerStatus.provider_unavailable.value: "Provider unavailable",
    AnswerStatus.rate_limited.value: "Rate limited",
    AnswerStatus.retrieval_unavailable.value: "Search unavailable",
}


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="data folder (default: RAGKB_DATA_DIR or ./data)")

    parser = argparse.ArgumentParser(
        prog="ragkb",
        description="rag-kb: a local knowledge base with cited answers written by your own assistant subscription.",
    )
    parser.add_argument("--version", action="version", version=f"rag-kb {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    serve = sub.add_parser("serve", parents=[common], help="run the web app and the API")
    serve.add_argument("--host", help="bind address (default: RAGKB_HOST or 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default: RAGKB_PORT or 8765)")
    serve.add_argument("--open", action="store_true", help="open the browser when the server is up")
    serve.add_argument("--reload", action="store_true", help="restart on code changes (development)")

    ingest = sub.add_parser("ingest", parents=[common], help="index files or folders into a collection")
    ingest.add_argument("paths", nargs="+", help="files or folders (folders are read recursively)")
    ingest.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"collection name, created when missing (default: {DEFAULT_COLLECTION})")

    ask = sub.add_parser("ask", parents=[common], help="ask a question and print the cited answer")
    ask.add_argument("question", help="the question, in quotes")
    ask.add_argument("--collection", help="collection name or id (default: the only one, or 'default')")
    ask.add_argument("--provider", choices=["auto", "claude_code", "codex", "extractive"], default="auto")
    ask.add_argument("--model", help="model for the provider, for example sonnet, opus or haiku")
    ask.add_argument("--top-k", type=int, dest="top_k", help="passages given to the composer")
    ask.add_argument("--json", action="store_true", help="print the answer contract as JSON")

    doctor = sub.add_parser("doctor", parents=[common], help="check providers, the embedding model and the stores")
    doctor.add_argument("--json", action="store_true", help="print the report as JSON")

    evaluate = sub.add_parser("eval", parents=[common], help="run the evaluation over a golden set")
    evaluate.add_argument("--golden", help="golden set file (default: samples/golden.yaml)")
    evaluate.add_argument("--live", action="store_true", help="use the real embedder and provider")
    evaluate.add_argument("--json", action="store_true", help="print the raw report as JSON")
    return parser


# ---------------------------------------------------------------------------- helpers


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "data_dir", None):
        overrides["DATA_DIR"] = Path(args.data_dir)
    return Settings(**overrides)


def _quiet_logs(settings: Settings) -> None:
    """Command output stays readable: only warnings and errors are logged, to stderr."""
    level = settings.LOG_LEVEL if settings.LOG_LEVEL.upper() in {"WARNING", "ERROR", "CRITICAL"} else "WARNING"
    configure_logging(level)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()
    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def _collect_files(paths: list[Path]) -> tuple[list[tuple[Path, str]], list[tuple[str, str]]]:
    """Expand folders recursively. Returns (file, display name) pairs and skipped (name, reason) pairs."""
    found: list[tuple[Path, str]] = []
    skipped: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for root in paths:
        if root.is_file():
            items = [(root, root.name)]
        else:
            items = []
            for p in sorted(root.rglob("*")):
                rel = p.relative_to(root)
                if any(part.startswith(".") for part in rel.parts) or not p.is_file():
                    continue
                items.append((p, str(rel).replace("\\", "/")))
        for path, display in items:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if supported_media_type(path.name) is None:
                if root.is_file():
                    skipped.append((display, "unsupported type"))
                continue
            found.append((path, display))
    return found, skipped


def _resolve_collection(deps: Deps, wanted: str | None) -> Any:
    meta = deps.meta
    collections = meta.list_collections()
    if wanted:
        by_id = meta.get_collection(wanted)
        return by_id or deps.find_collection_by_name(wanted)
    if len(collections) == 1:
        return collections[0]
    return deps.find_collection_by_name(DEFAULT_COLLECTION)


def _missing_parts(deps: Deps, *names: str) -> list[str]:
    return [name for name in names if getattr(deps, name, None) is None]


def _print_problems(deps: Deps) -> None:
    for name, reason in sorted(deps.problems.items()):
        print(f"  {name}: {reason}", file=sys.stderr)


def _short(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"


def format_answer(answer: Answer) -> str:
    """The answer as plain text, then a numbered sources list."""
    status = answer.status.value if isinstance(answer.status, AnswerStatus) else str(answer.status)
    lines: list[str] = []
    if status == AnswerStatus.answered.value and answer.answer:
        lines.append(answer.answer.strip())
    else:
        lines.append(f"{_STATE_LABELS.get(status, status)}: {answer.message or ''}".rstrip())
    if answer.citations:
        lines.append("")
        lines.append("Sources:")
        for c in answer.citations:
            where = [f"p. {c.page}"] if c.page is not None else []
            if c.section:
                where.append(c.section)
            suffix = f" ({', '.join(where)})" if where else ""
            lines.append(f"  [{c.marker}] {c.title}{suffix}  score {_short(c.score)}")
    info = []
    if answer.provider:
        model = f" ({answer.provider.model})" if answer.provider.model else ""
        info.append(f"{answer.provider.name}{model}")
        if answer.provider.elapsed_ms:
            info.append(f"{answer.provider.elapsed_ms / 1000:.1f} s")
    info.append(f"{answer.passages_considered} passages considered")
    if answer.knowledge_version:
        info.append(f"knowledge {answer.knowledge_version}")
    lines.append("")
    lines.append(", ".join(info))
    return "\n".join(lines)


# ---------------------------------------------------------------------------- commands


def _open_when_ready(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=2):
                break
        except Exception:  # noqa: BLE001 - not up yet, try again
            time.sleep(0.5)
    webbrowser.open(url)


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _settings(args)
    host = args.host or settings.HOST
    port = args.port or settings.PORT
    if not 0 < port < 65536:
        print("The port must be between 1 and 65535.", file=sys.stderr)
        return EXIT_USAGE
    if args.data_dir:
        # The app factory (and reload workers) read settings from the environment.
        os.environ["RAGKB_DATA_DIR"] = str(Path(args.data_dir).resolve())
    settings.ensure_dirs()
    browse_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    url = f"http://{browse_host}:{port}/"
    print(f"rag-kb {__version__} at {url}  (data: {settings.DATA_DIR.resolve()})", flush=True)
    print(f"MCP endpoint: {url}mcp" if settings.MCP_ENABLED else "MCP endpoint: disabled", flush=True)
    if not settings.API_PUBLIC and not host_is_loopback(host):
        print("The API and the web app answer only on this machine; set RAGKB_API_PUBLIC=true to open them.",
              flush=True)
    if args.open:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(
        "ragkb.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=bool(args.reload),
        access_log=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(missing)}", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings(args)
    _quiet_logs(settings)
    deps = build_deps(settings)
    try:
        absent = _missing_parts(deps, "meta", "pipeline")
        if absent:
            print("Ingestion is not available in this installation:", file=sys.stderr)
            _print_problems(deps)
            return EXIT_FAIL
        files, skipped = _collect_files(paths)
        if not files:
            print(f"No supported files found. Supported types: {', '.join(supported_types())}.", file=sys.stderr)
            return EXIT_FAIL
        collection = deps.find_collection_by_name(args.collection)
        if collection is None:
            collection = deps.create_collection(args.collection)
            print(f"Created collection '{collection.name}'.")
        cap = settings.MAX_UPLOAD_MB * 1024 * 1024
        rows: list[list[str]] = []
        counts = {"ready": 0, "already indexed": 0, "failed": 0, "skipped": len(skipped)}
        for path, display in files:
            print(f"  indexing {display} ...", end="\r", flush=True)
            data = path.read_bytes()
            if len(data) > cap:
                rows.append([display, "failed", "", "", f"larger than {settings.MAX_UPLOAD_MB} MB"])
                counts["failed"] += 1
                continue
            if not data:
                rows.append([display, "failed", "", "", "empty file"])
                counts["failed"] += 1
                continue
            existing = deps.meta.find_by_hash(collection.id, hashlib.sha256(data).hexdigest())
            if existing is not None and existing.status == DocumentStatus.ready:
                rows.append([display, "already indexed", str(existing.chunk_count), str(existing.pages or ""), ""])
                counts["already indexed"] += 1
                continue
            try:
                doc = deps.pipeline.ingest_bytes(collection.id, data, path.name)
            except Exception as exc:  # noqa: BLE001 - each check reports its own failure
                rows.append([display, "failed", "", "", plain_error(exc, "could not be indexed")])
                counts["failed"] += 1
                continue
            status = doc.status.value if isinstance(doc.status, DocumentStatus) else str(doc.status)
            if status == DocumentStatus.failed.value:
                counts["failed"] += 1
            else:
                counts["ready"] += 1
            rows.append([display, status, str(doc.chunk_count), str(doc.pages or ""), doc.error or ""])
        rows.extend([name, "skipped", "", "", reason] for name, reason in skipped)
        print(" " * 60, end="\r")
        print(_table(["File", "Status", "Chunks", "Pages", "Note"], rows))
        print()
        print(f"{counts['ready']} indexed, {counts['already indexed']} already indexed, {counts['failed']} failed, "
              f"{counts['skipped']} skipped. Collection '{collection.name}', knowledge {deps.knowledge_version()}.")
        return EXIT_FAIL if counts["failed"] else EXIT_OK
    finally:
        deps.close()


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.question.strip():
        print("The question is empty.", file=sys.stderr)
        return EXIT_USAGE
    if args.top_k is not None and not 1 <= args.top_k <= 20:
        print("--top-k must be between 1 and 20.", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings(args)
    _quiet_logs(settings)
    deps = build_deps(settings)
    try:
        if deps.meta is None:
            print("The knowledge base is not available in this installation:", file=sys.stderr)
            _print_problems(deps)
            return EXIT_FAIL
        collection = _resolve_collection(deps, args.collection)
        if collection is None:
            names = ", ".join(c.name for c in deps.meta.list_collections()) or "none yet"
            print(f"Collection not found. Use --collection with one of: {names}.", file=sys.stderr)
            return EXIT_FAIL
        try:
            answer = deps.ask(collection.id, args.question.strip(), top_k=args.top_k, provider=args.provider,
                              model=args.model)
        except ApiError as exc:
            print(exc.message, file=sys.stderr)
            return EXIT_FAIL
        print(answer.model_dump_json(indent=2) if args.json else format_answer(answer))
        ok = {AnswerStatus.answered.value, AnswerStatus.no_answer.value}
        status = answer.status.value if isinstance(answer.status, AnswerStatus) else str(answer.status)
        return EXIT_OK if status in ok else EXIT_FAIL
    finally:
        deps.close()


def _check(checks: list[dict[str, Any]], name: str, ok: bool | None, detail: str, essential: bool = True) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail, "essential": essential})


def _data_dir_check(settings: Settings) -> tuple[bool, str]:
    folder = settings.DATA_DIR.resolve()
    try:
        settings.ensure_dirs()
        probe = folder / ".ragkb-write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, f"{folder} (writable)"
    except OSError:
        return False, f"{folder} (not writable)"


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _quiet_logs(settings)
    checks: list[dict[str, Any]] = []
    py_ok = sys.version_info >= (3, 11)
    _check(checks, "Python", py_ok, f"{platform.python_version()} ({'3.11 or newer' if py_ok else 'needs 3.11 or newer'})")
    _check(checks, "rag-kb", True, __version__, essential=False)
    data_ok, data_detail = _data_dir_check(settings)
    _check(checks, "Data folder", data_ok, data_detail)
    if not args.json:
        print("Checking (the first run downloads the embedding model, about 130 MB) ...", file=sys.stderr)
    deps = build_deps(settings)
    providers: list[dict[str, Any]] = []
    provider_report: dict[str, Any] = {}
    try:
        for label in sorted(missing_parts()):
            _check(checks, f"Part: {label}", False, "not installed", essential=False)

        if deps.meta is not None:
            try:
                n = len(deps.meta.list_collections())
                version = deps.knowledge_version() or "none yet"
                _check(checks, "Metadata store", True, f"ok, {n} collections, knowledge {version}")
            except Exception as exc:  # noqa: BLE001 - each check reports its own failure
                _check(checks, "Metadata store", False, plain_error(exc, "could not be read"))
        else:
            _check(checks, "Metadata store", False, deps.problems.get("metadata store", "not available"))

        if deps.store is not None:
            try:
                _check(checks, "Vector store", True, f"ok, {int(deps.store.count())} passages")
            except Exception as exc:  # noqa: BLE001 - each check reports its own failure
                _check(checks, "Vector store", False, plain_error(exc, "could not be read"))
        else:
            _check(checks, "Vector store", False, deps.problems.get("vector store", "not available"))

        if deps.embedder is not None:
            started = time.perf_counter()
            try:
                vector = deps.embedder.embed_query("rag-kb doctor check")
                ms = int((time.perf_counter() - started) * 1000)
                kind = "fake embedder (tests)" if deps.fake_embedder else getattr(deps.embedder, "name", "embedder")
                _check(checks, "Embedding model", len(vector) == deps.embedder.dims,
                       f"{kind}, {deps.embedder.dims} dimensions, test embedding in {ms} ms")
            except Exception as exc:  # noqa: BLE001 - each check reports its own failure
                _check(checks, "Embedding model", False, plain_error(exc, "could not embed a test sentence"))
        else:
            name = "fake embedder" if fake_embedder_requested() else settings.EMBED_MODEL
            _check(checks, "Embedding model", False, f"{name}: {deps.problems.get('embedder', 'not available')}")

        reranker_ok = deps.reranker is not None or settings.RERANKER == "none"
        _check(checks, "Reranker", reranker_ok,
               settings.RERANKER if settings.RERANKER == "none" else f"local ({settings.RERANK_MODEL})",
               essential=False)

        if deps.providers is not None:
            try:
                for status in deps.providers.statuses():
                    providers.append(status.model_dump())
                usable = [p for p in providers if p["installed"] and p["signed_in"]]
                try:
                    chosen = deps.providers.select(None if settings.PROVIDER == "auto" else settings.PROVIDER)
                    choice = provider_label(getattr(chosen, "name", None))
                except ProviderError as exc:
                    choice = f"none ({exc.kind})"
                _check(checks, "Answer providers", bool(usable),
                       f"{len(usable)} of {len(providers)} ready; answers will use {choice}")
                provider_report = deps.providers.doctor_report()
            except Exception as exc:  # noqa: BLE001 - each check reports its own failure
                _check(checks, "Answer providers", False, plain_error(exc, "status could not be checked"))
        else:
            _check(checks, "Answer providers", False, deps.problems.get("providers", "not available"))

        _check(checks, "MCP server", None if not settings.MCP_ENABLED else "mcp" not in deps.problems,
               "enabled at /mcp" if settings.MCP_ENABLED else "disabled", essential=False)
    finally:
        deps.close()

    ok = all(c["ok"] for c in checks if c["essential"])
    if args.json:
        print(json.dumps({"ok": ok, "checks": checks, "providers": providers, "provider_report": provider_report},
                         indent=2, default=str))
        return EXIT_OK if ok else EXIT_FAIL

    marks = {True: "ok", False: "FAIL", None: "--"}
    for c in checks:
        mark = marks[c["ok"]] if c["essential"] or c["ok"] is not False else "warn"
        print(f"[{mark:>4}] {c['name']}: {c['detail']}")
    if providers:
        print()
        rows = []
        for p in providers:
            state = "ready" if p["installed"] and p["signed_in"] else ("not signed in" if p["installed"] else "not installed")
            rows.append([provider_label(p["name"]), state, p.get("version") or "", p.get("fix_command") or ""])
        print(_table(["Provider", "State", "Version", "Fix"], rows))
    print()
    print("All essential checks passed." if ok else "Some essential checks failed (marked FAIL).")
    return EXIT_OK if ok else EXIT_FAIL


def _default_golden() -> Path | None:
    for candidate in (Path.cwd() / "samples" / "golden.yaml",
                      Path(__file__).resolve().parents[1] / "samples" / "golden.yaml"):
        if candidate.is_file():
            return candidate
    return None


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the golden set. Exit code from the gate: 0 PASS, 1 FAIL, 2 SKIPPED (too little could run)."""
    golden = Path(args.golden) if args.golden else _default_golden()
    if golden is None or not golden.is_file():
        print("Golden set not found. Pass --golden with the path to a golden set file.", file=sys.stderr)
        return EXIT_USAGE
    if run_eval is None or format_report is None:
        print("The evaluation part is not installed.", file=sys.stderr)
        return EXIT_FAIL
    settings = _settings(args)
    _quiet_logs(settings)
    try:
        report = run_eval(golden, settings, live=bool(args.live))
    except Exception as exc:  # noqa: BLE001 - a failed run is reported, not raised
        print(f"The evaluation could not run: {plain_error(exc, 'unexpected error (see the log)')}", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    return int(report["gate"]["exit_code"])


COMMANDS = {"serve": cmd_serve, "ingest": cmd_ingest, "ask": cmd_ask, "doctor": cmd_doctor, "eval": cmd_eval}


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
