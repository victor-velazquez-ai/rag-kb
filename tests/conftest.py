"""Shared test fixtures: a temporary data folder, the real stores on it, the fake embedder and the pipeline.

Everything is offline: FakeEmbedder (deterministic hashed bag of words) stands in for the model, and the
reranker (on by default, a 90 MB download) is off. A test that needs a real model is live: it runs only
with RAGKB_LIVE_TESTS=1.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ragkb.config import Settings
from ragkb.ingest.parsers import SUPPORTED_SUFFIXES
from ragkb.ingest.pipeline import IngestPipeline
from ragkb.models import Collection
from ragkb.retrieval import reranker as reranker_module
from ragkb.retrieval.embedder import FakeEmbedder
from ragkb.store.lancedb_store import LanceDBStore
from ragkb.store.sqlite_meta import SQLiteMeta

SAMPLE_CORPUS = Path(__file__).resolve().parents[1] / "samples" / "corpus"
LIVE_TESTS = os.environ.get("RAGKB_LIVE_TESTS") == "1"


@pytest.fixture(autouse=True)
def no_reranker_download(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test runs with RAGKB_RERANKER=none unless it sets the variable itself. Outside live runs, a
    test that still tries to load the real cross-encoder fails, even though the reranker would quietly
    fall back to the fused order: a test that stands in for the model patches TextCrossEncoder itself."""
    monkeypatch.setenv("RAGKB_RERANKER", "none")
    attempts: list[str] = []
    if not LIVE_TESTS:

        def refuse(model_name: str, *args: object, **kwargs: object) -> None:
            attempts.append(model_name)
            raise RuntimeError("offline tests do not load the reranker model")

        monkeypatch.setattr(reranker_module, "TextCrossEncoder", refuse)
    yield
    assert not attempts, f"an offline test tried to load the reranker model {attempts}; mark it live"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """A minimal valid PDF: one Helvetica text line per list item; an empty page list item is a page
    with no text at all (what a scanned page looks like to a text extractor)."""
    objects: dict[int, bytes] = {}
    kids: list[str] = []
    for i, lines in enumerate(pages):
        page_id, content_id = 4 + 2 * i, 5 + 2 * i
        kids.append(f"{page_id} 0 R")
        ops = ["BT", "/F1 11 Tf"]
        y = 750
        for line in lines:
            if line:
                ops.append(f"1 0 0 1 72 {y} Tm ({_pdf_escape(line)}) Tj")
            y -= 16
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    info_id = 4 + 2 * len(pages)
    objects[info_id] = f"<< /Title ({_pdf_escape(title or '')}) >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def pdf_builder() -> Callable[..., bytes]:
    return build_pdf


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(_env_file=None, DATA_DIR=tmp_path / "data", CHUNK_TOKENS=350, CHUNK_OVERLAP_TOKENS=60,
                 RERANKER="none")
    s.ensure_dirs()
    return s


@pytest.fixture
def meta(settings: Settings) -> SQLiteMeta:
    return SQLiteMeta(settings.meta_path)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dims=384)


@pytest.fixture
def store(settings: Settings, embedder: FakeEmbedder) -> LanceDBStore:
    return LanceDBStore(settings.lancedb_dir, dims=embedder.dims)


@pytest.fixture
def pipeline(
    settings: Settings, meta: SQLiteMeta, store: LanceDBStore, embedder: FakeEmbedder
) -> IngestPipeline:
    return IngestPipeline(settings, meta, store, embedder)


@pytest.fixture
def collection(meta: SQLiteMeta) -> Collection:
    return meta.create_collection("Test collection", "Created by a test")


@pytest.fixture
def sample_corpus_dir() -> Path:
    return SAMPLE_CORPUS


@pytest.fixture
def indexed_corpus(pipeline: IngestPipeline, collection: Collection, sample_corpus_dir: Path) -> Collection:
    """The whole sample corpus ingested into ``collection``."""
    for path in sorted(sample_corpus_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = pipeline.ingest_file(collection.id, path).document
            assert doc.status.value == "ready", (path.name, doc.error)
    return collection
