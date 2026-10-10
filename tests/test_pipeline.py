"""Ingestion pipeline: status, dedupe by hash, replace by title, durability on failure, versions."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from ragkb.config import Settings
from ragkb.ingest.parsers import EmptyDocumentError, UnsupportedFileTypeError
from ragkb.ingest.pipeline import IngestPipeline, NotFoundError, safe_filename
from ragkb.models import Collection, DocumentStatus
from ragkb.retrieval.embedder import EmbedderError, FakeEmbedder
from ragkb.store.lancedb_store import LanceDBStore
from ragkb.store.sqlite_meta import SQLiteMeta

VERSION = re.compile(r"v\d{8}\.\d+")


def markdown(title: str, sections: dict[str, str]) -> bytes:
    parts = [f"# {title}", ""]
    for heading, body in sections.items():
        parts += [f"## {heading}", "", body, ""]
    return "\n".join(parts).encode("utf-8")


def filler(topic: str, n: int = 12) -> str:
    return " ".join(
        f"Rule {i} about {topic} applies to every team and every office location." for i in range(n)
    )


V1 = markdown(
    "Travel Rules",
    {
        "Flights": "Economy class is required for flights under six hours. " + filler("flights"),
        "Hotels": "Hotels in capital cities are capped at 210 euros per night. " + filler("hotels"),
        "Meals": "The zebrafruit allowance is 70 euros per day abroad. " + filler("meals"),
        "Trains": "Trains are preferred for journeys under 500 kilometres. " + filler("trains"),
        "Cars": "Private cars are reimbursed at 30 cents per kilometre. " + filler("cars"),
    },
)
V2 = markdown(
    "Travel Rules", {"Flights": "Premium economy is allowed for flights over six hours. " + filler("flights")}
)


class FlakyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dims=384)
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise EmbedderError("The embedding model fake failed: simulated outage")
        return super().embed(texts)


@pytest.fixture
def flaky() -> FlakyEmbedder:
    return FlakyEmbedder()


@pytest.fixture
def flaky_pipeline(
    settings: Settings, meta: SQLiteMeta, store: LanceDBStore, flaky: FlakyEmbedder
) -> IngestPipeline:
    return IngestPipeline(settings, meta, store, flaky)


def texts(store: LanceDBStore, doc_id: str) -> str:
    return "\n".join(c.text for c in store.chunks_for(doc_id))


# ------------------------------------------------------------------------------------------- ingest


def test_ingest_bytes_indexes_a_new_document(
    pipeline: IngestPipeline,
    meta: SQLiteMeta,
    store: LanceDBStore,
    settings: Settings,
    collection: Collection,
) -> None:
    doc = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    assert doc.status == DocumentStatus.ready and doc.error is None
    assert doc.title == "Travel Rules" and doc.filename == "travel.md" and doc.media_type == "text/markdown"
    assert doc.size_bytes == len(V1) and doc.content_hash == hashlib.sha256(V1).hexdigest()
    assert doc.pages is None and doc.indexed_at is not None
    assert VERSION.fullmatch(doc.knowledge_version or "")
    chunks = store.chunks_for(doc.id)
    assert doc.chunk_count == len(chunks) >= 5
    assert [c.id for c in chunks] == [f"{doc.id}#{i}" for i in range(len(chunks))]
    assert {c.knowledge_version for c in chunks} == {doc.knowledge_version}
    assert {c.title for c in chunks} == {"Travel Rules"} and {c.collection_id for c in chunks} == {
        collection.id
    }
    assert chunks[1].section == "Hotels" and chunks[1].embed_text.startswith("Travel Rules > Hotels: ")
    kept = settings.uploads_dir / collection.id / doc.id / "travel.md"
    assert kept.read_bytes() == V1 and pipeline.original_path(doc) == kept
    staging = settings.uploads_dir / collection.id / ".staging"
    assert not staging.exists() or not any(staging.iterdir())
    assert meta.get_document(doc.id) == doc
    audit = meta.list_audit(event="ingest")[0]
    assert audit["outcome"] == "indexed" and audit["doc_id"] == doc.id and audit["chunks"] == len(chunks)


def test_same_bytes_are_already_indexed(
    pipeline: IngestPipeline, meta: SQLiteMeta, collection: Collection
) -> None:
    first = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    version = meta.current_version()
    again = pipeline.ingest_bytes(collection.id, V1, "renamed-copy.md", title="Another title")
    assert again == first
    assert meta.current_version() == version, "no work, no version bump"
    assert len(meta.list_documents(collection.id)) == 1
    assert meta.list_audit(event="ingest")[0]["outcome"] == "duplicate"
    assert pipeline.find_duplicate(collection.id, V1) == first
    assert pipeline.find_duplicate(collection.id, V2) is None


def test_same_bytes_in_another_collection_is_a_new_document(
    pipeline: IngestPipeline, meta: SQLiteMeta, collection: Collection
) -> None:
    other = meta.create_collection("Other")
    first = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    second = pipeline.ingest_bytes(other.id, V1, "travel.md")
    assert second.id != first.id and second.collection_id == other.id


def test_same_title_replaces_and_drops_the_stale_tail(
    pipeline: IngestPipeline, meta: SQLiteMeta, store: LanceDBStore, collection: Collection
) -> None:
    v1 = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    v2 = pipeline.ingest_bytes(collection.id, V2, "travel.md")
    assert v2.id == v1.id and v2.status == DocumentStatus.ready
    assert v2.chunk_count == len(store.chunks_for(v2.id)) < v1.chunk_count
    assert v2.content_hash == hashlib.sha256(V2).hexdigest() and v2.size_bytes == len(V2)
    assert v2.knowledge_version != v1.knowledge_version
    assert "Premium economy" in texts(store, v2.id) and "zebrafruit" not in texts(store, v2.id)
    assert store.text_search(collection.id, "zebrafruit", 5) == []
    assert [h.chunk.doc_id for h in store.text_search(collection.id, "premium", 5)] == [v2.id]
    assert len(meta.list_documents(collection.id)) == 1
    assert meta.list_audit(event="ingest")[0]["outcome"] == "replaced"


def test_explicit_title_replaces_across_file_names_and_types(
    pipeline: IngestPipeline, settings: Settings, collection: Collection
) -> None:
    first = pipeline.ingest_bytes(collection.id, V1, "travel.md", title="Travel Policy")
    second = pipeline.ingest_bytes(
        collection.id, b"Travel Policy v3\n\nAll travel is suspended.\n", "t.txt", title="travel policy"
    )
    assert second.id == first.id
    assert (second.filename, second.media_type, second.title) == ("t.txt", "text/plain", "travel policy")
    folder = settings.uploads_dir / collection.id / first.id
    assert [p.name for p in folder.iterdir()] == ["t.txt"], "the older original is removed"


# --------------------------------------------------------------------------------------- durability


def test_failed_replacement_keeps_the_previous_version(
    flaky_pipeline: IngestPipeline,
    flaky: FlakyEmbedder,
    meta: SQLiteMeta,
    store: LanceDBStore,
    settings: Settings,
    collection: Collection,
) -> None:
    v1 = flaky_pipeline.ingest_bytes(collection.id, V1, "travel.md")
    version = meta.current_version()
    flaky.fail = True
    failed = flaky_pipeline.ingest_bytes(collection.id, V2, "travel-v2.md")
    assert failed.id == v1.id and failed.status == DocumentStatus.failed
    assert "simulated outage" in failed.error and "previous version is still searchable" in failed.error
    assert (failed.chunk_count, failed.content_hash, failed.filename) == (
        v1.chunk_count,
        v1.content_hash,
        "travel.md",
    )
    assert meta.current_version() == version
    assert len(store.chunks_for(v1.id)) == v1.chunk_count
    assert [h.chunk.doc_id for h in store.text_search(collection.id, "zebrafruit", 5)] == [v1.id]
    assert (settings.uploads_dir / collection.id / v1.id / "travel.md").read_bytes() == V1
    assert not (settings.uploads_dir / collection.id / v1.id / "travel-v2.md").exists()
    assert meta.list_audit(event="ingest")[0]["outcome"] == "failed"

    flaky.fail = False
    fixed = flaky_pipeline.ingest_bytes(collection.id, V2, "travel-v2.md")
    assert fixed.id == v1.id and fixed.status == DocumentStatus.ready and fixed.error is None
    assert "Premium economy" in texts(store, v1.id) and "zebrafruit" not in texts(store, v1.id)
    assert [p.name for p in (settings.uploads_dir / collection.id / v1.id).iterdir()] == ["travel-v2.md"]


def test_unreadable_replacement_keeps_the_previous_version(
    pipeline: IngestPipeline, store: LanceDBStore, collection: Collection, pdf_builder
) -> None:
    v1 = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    failed = pipeline.ingest_bytes(collection.id, pdf_builder([[], []]), "scan.pdf", title="Travel Rules")
    assert failed.id == v1.id and failed.status == DocumentStatus.failed
    assert "scanned" in failed.error
    assert len(store.chunks_for(v1.id)) == v1.chunk_count
    assert store.text_search(collection.id, "zebrafruit", 5)


def test_new_document_failure_is_recorded(
    pipeline: IngestPipeline,
    meta: SQLiteMeta,
    store: LanceDBStore,
    settings: Settings,
    collection: Collection,
    pdf_builder,
) -> None:
    doc = pipeline.ingest_bytes(collection.id, pdf_builder([[], [], []]), "scan.pdf")
    assert doc.status == DocumentStatus.failed and doc.chunk_count == 0
    assert doc.error.startswith("This PDF has no extractable text")
    assert doc.title == "Scan"
    assert store.chunks_for(doc.id) == []
    assert (settings.uploads_dir / collection.id / doc.id / "scan.pdf").exists(), (
        "kept for download and retry"
    )
    assert meta.list_audit(event="ingest")[0]["outcome"] == "failed"


def test_retry_after_failure_reuses_the_document(
    flaky_pipeline: IngestPipeline, flaky: FlakyEmbedder, meta: SQLiteMeta, collection: Collection
) -> None:
    flaky.fail = True
    failed = flaky_pipeline.ingest_bytes(collection.id, V1, "travel.md")
    assert failed.status == DocumentStatus.failed and failed.chunk_count == 0
    assert "previous version" not in failed.error
    flaky.fail = False
    retried = flaky_pipeline.ingest_bytes(collection.id, V1, "travel.md")
    assert retried.id == failed.id and retried.status == DocumentStatus.ready
    assert len(meta.list_documents(collection.id)) == 1


def test_errors_raised_before_any_work(
    pipeline: IngestPipeline, meta: SQLiteMeta, settings: Settings, collection: Collection
) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        pipeline.ingest_bytes(collection.id, b"MZ", "tool.exe")
    with pytest.raises(EmptyDocumentError):
        pipeline.ingest_bytes(collection.id, b"", "empty.md")
    with pytest.raises(NotFoundError):
        pipeline.ingest_bytes("no-such-collection", V1, "travel.md")
    assert meta.list_documents(collection.id) == []
    assert meta.current_version() == ""


def test_ingest_file_copies_the_original(
    pipeline: IngestPipeline, collection: Collection, sample_corpus_dir: Path
) -> None:
    source = sample_corpus_dir / "employee-handbook.md"
    before = source.read_bytes()
    doc = pipeline.ingest_file(collection.id, source)
    assert doc.status == DocumentStatus.ready and doc.title == "Employee Handbook"
    assert pipeline.original_path(doc).read_bytes() == before
    assert source.read_bytes() == before


# ------------------------------------------------------------------------------ reindex and delete


def test_reindex_from_the_kept_original(
    pipeline: IngestPipeline, meta: SQLiteMeta, store: LanceDBStore, collection: Collection
) -> None:
    doc = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    ids = [c.id for c in store.chunks_for(doc.id)]
    again = pipeline.reindex(doc.id)
    assert again.id == doc.id and again.status == DocumentStatus.ready
    assert again.knowledge_version != doc.knowledge_version
    assert [c.id for c in store.chunks_for(doc.id)] == ids
    assert {c.knowledge_version for c in store.chunks_for(doc.id)} == {again.knowledge_version}
    assert meta.list_audit(event="reindex")[0]["outcome"] == "indexed"
    with pytest.raises(NotFoundError):
        pipeline.reindex("missing")


def test_reindex_without_the_original_keeps_the_index(
    pipeline: IngestPipeline, store: LanceDBStore, collection: Collection
) -> None:
    doc = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    pipeline.original_path(doc).unlink()
    failed = pipeline.reindex(doc.id)
    assert failed.status == DocumentStatus.failed and "original file is missing" in failed.error
    assert "previous index is still searchable" in failed.error
    assert len(store.chunks_for(doc.id)) == doc.chunk_count


def test_delete_removes_chunks_metadata_and_original(
    pipeline: IngestPipeline,
    meta: SQLiteMeta,
    store: LanceDBStore,
    settings: Settings,
    collection: Collection,
) -> None:
    keep = pipeline.ingest_bytes(collection.id, V2, "keep.md", title="Keep me")
    gone = pipeline.ingest_bytes(collection.id, V1, "travel.md")
    version = meta.current_version()
    pipeline.delete(gone.id)
    assert store.chunks_for(gone.id) == [] and meta.get_document(gone.id) is None
    assert not (settings.uploads_dir / collection.id / gone.id).exists()
    assert len(store.chunks_for(keep.id)) == keep.chunk_count and meta.get_document(keep.id) is not None
    assert meta.current_version() != version
    audit = meta.list_audit(event="delete")[0]
    assert audit["doc_id"] == gone.id and audit["chunks"] == gone.chunk_count
    with pytest.raises(NotFoundError):
        pipeline.delete(gone.id)


def test_delete_collection_removes_only_that_collection(
    pipeline: IngestPipeline,
    meta: SQLiteMeta,
    store: LanceDBStore,
    settings: Settings,
    collection: Collection,
) -> None:
    other = meta.create_collection("Other")
    pipeline.ingest_bytes(collection.id, V1, "travel.md")
    survivor = pipeline.ingest_bytes(other.id, V2, "travel.md")
    removed = pipeline.delete_collection(collection.id)
    assert removed > 0 and store.count(collection.id) == 0
    assert meta.get_collection(collection.id) is None and meta.list_documents(collection.id) == []
    assert not (settings.uploads_dir / collection.id).exists()
    assert store.count(other.id) == survivor.chunk_count
    with pytest.raises(NotFoundError):
        pipeline.delete_collection(collection.id)


# ------------------------------------------------------------------------------- versions and names


def test_every_change_bumps_the_version(
    pipeline: IngestPipeline, meta: SQLiteMeta, collection: Collection
) -> None:
    a = pipeline.ingest_bytes(collection.id, V1, "a.md", title="A")
    b = pipeline.ingest_bytes(collection.id, V2, "b.md", title="B")
    day, n = a.knowledge_version.split(".")
    assert VERSION.fullmatch(a.knowledge_version) and b.knowledge_version == f"{day}.{int(n) + 1}"
    pipeline.delete(a.id)
    assert meta.current_version() == f"{day}.{int(n) + 2}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.md", "passwd.md"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ('bad<>:"|?*name.md', "bad_______name.md"),
        ("CON.txt", "_CON.txt"),
        (".md", "upload.md"),
        ("trailing. ", "trailing"),
        ("", "upload"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_path_traversal_upload_stays_inside_the_uploads_folder(
    pipeline: IngestPipeline, settings: Settings, collection: Collection
) -> None:
    doc = pipeline.ingest_bytes(collection.id, V1, "../../../outside.md")
    path = pipeline.original_path(doc).resolve()
    assert settings.uploads_dir.resolve() in path.parents and path.name == "outside.md"


def test_concurrent_ingestion_is_serialized_and_consistent(
    pipeline: IngestPipeline, meta: SQLiteMeta, collection: Collection
) -> None:
    results = {}
    errors: list[BaseException] = []

    def ingest(n: int) -> None:
        try:
            data = markdown(f"Doc {n}", {"Body": f"Unique body {n}. " + filler(f"topic {n}", 3)})
            results[n] = pipeline.ingest_bytes(collection.id, data, f"doc{n}.md")
        except BaseException as exc:  # noqa: BLE001 - report any failure from the thread
            errors.append(exc)

    threads = [threading.Thread(target=ingest, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(d.status == DocumentStatus.ready for d in results.values())
    assert len({d.knowledge_version for d in results.values()}) == 4
    assert len(meta.list_documents(collection.id)) == 4


def test_the_whole_sample_corpus_indexes(
    indexed_corpus: Collection, meta: SQLiteMeta, store: LanceDBStore
) -> None:
    docs = meta.list_documents(indexed_corpus.id)
    assert len(docs) >= 8
    assert all(d.status == DocumentStatus.ready and d.chunk_count > 0 for d in docs)
    assert store.count(indexed_corpus.id) == sum(d.chunk_count for d in docs)
    pdf = next(d for d in docs if d.media_type == "application/pdf")
    assert pdf.pages == 2
