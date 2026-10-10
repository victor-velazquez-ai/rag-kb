"""Stores: LanceDB chunks (search, upsert, stale tail, isolation) and SQLite metadata (CRUD, version)."""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragkb.models import Chunk, Document, DocumentStatus, Feedback
from ragkb.retrieval.embedder import FakeEmbedder
from ragkb.store import sqlite_meta
from ragkb.store.lancedb_store import OPTIMIZE_EVERY, LanceDBStore, StoreError
from ragkb.store.sqlite_meta import SQLiteMeta

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def make_chunks(
    doc_id: str, collection_id: str, texts: list[str], version: str = "v20261010.1"
) -> list[Chunk]:
    return [
        Chunk(
            id=f"{doc_id}#{i}",
            doc_id=doc_id,
            collection_id=collection_id,
            ordinal=i,
            text=text,
            embed_text=f"Doc {doc_id}: {text}",
            title=f"Doc {doc_id}",
            section="Section" if i % 2 else "",
            page=i + 1 if i % 3 else None,
            char_start=i * 10,
            char_end=i * 10 + len(text),
            token_estimate=len(text.split()),
            knowledge_version=version,
            created_at=NOW,
        )
        for i, text in enumerate(texts)
    ]


def put(store: LanceDBStore, embedder: FakeEmbedder, chunks: list[Chunk]) -> None:
    store.upsert(chunks, embedder.embed([c.embed_text for c in chunks]))


TRAVEL = [
    "Economy class is required for flights shorter than six hours.",
    "Hotel stays in capital cities are capped at 210 euros per night.",
    "The daily allowance for international travel is 70 euros.",
    "Expense claims must be filed within thirty days.",
]
SECURITY = [
    "Passwords must contain at least fourteen characters.",
    "Laptops lock after five minutes without activity.",
    "Report lost laptops to the service desk within one hour.",
]


# ============================================================================================ LanceDB


def test_empty_store_answers_without_a_table(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    assert store.count() == 0
    assert store.count("c1") == 0
    assert store.has_documents("c1") is False
    assert store.vector_search("c1", embedder.embed_query("x"), 5) == []
    assert store.text_search("c1", "x", 5) == []
    assert store.chunks_for("d1") == []
    assert store.delete_doc("d1") == 0
    assert store.delete_collection("c1") == 0
    store.upsert([], [])
    assert store.count() == 0


def test_upsert_and_vector_search(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    hits = store.vector_search("c1", embedder.embed_query("hotel capital cities night cap"), 4)
    assert hits[0].chunk.id == "d1#1"
    assert hits[0].vector_score == hits[0].score
    assert all(0.0 <= h.vector_score <= 1.0001 for h in hits)
    assert [h.vector_score for h in hits] == sorted((h.vector_score for h in hits), reverse=True)
    chunk = hits[0].chunk
    assert chunk.page == 2 and chunk.section == "Section" and chunk.knowledge_version == "v20261010.1"
    assert chunk.created_at == NOW


def test_vector_search_applies_the_minimum_score(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    query = embedder.embed_query("hotel capital cities night cap")
    everything = store.vector_search("c1", query, 10)
    floor = everything[0].vector_score - 0.0001
    filtered = store.vector_search("c1", query, 10, min_score=floor)
    assert [h.chunk.id for h in filtered] == [everything[0].chunk.id]


def test_text_search_ranks_keyword_matches(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    hits = store.text_search("c1", "allowance", 5)
    assert [h.chunk.id for h in hits] == ["d1#2"]
    assert hits[0].lexical_rank == 1 and hits[0].vector_score is None and hits[0].score > 0
    assert store.text_search("c1", "   ", 5) == []
    assert store.text_search("c1", "nothing-matches-this-word", 5) == []


@pytest.mark.parametrize("query", ["what's the per-diem?", 'he said "hello', "AND OR NOT", "a:b (c", "*"])
def test_text_search_survives_punctuation(store: LanceDBStore, embedder: FakeEmbedder, query: str) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    store.text_search("c1", query, 5)


def test_upsert_replaces_by_id_and_refreshes_the_text_index(
    store: LanceDBStore, embedder: FakeEmbedder
) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    replacement = make_chunks(
        "d1", "c1", ["Rail travel is preferred under 500 kilometres."], version="v20261010.2"
    )
    put(store, embedder, replacement)
    assert store.count("c1") == len(TRAVEL)
    assert store.chunks_for("d1")[0].text.startswith("Rail travel")
    assert store.chunks_for("d1")[0].knowledge_version == "v20261010.2"
    assert [h.chunk.id for h in store.text_search("c1", "kilometres", 5)] == ["d1#0"]
    assert store.text_search("c1", "economy", 5) == []


def test_text_index_is_refreshed_periodically_and_never_misses_rows(
    store: LanceDBStore, embedder: FakeEmbedder
) -> None:
    for n in range(OPTIMIZE_EVERY + 3):
        put(store, embedder, make_chunks(f"d{n}", "c1", [f"Unique marker word{n} in document {n}."]))
        assert [h.chunk.id for h in store.text_search("c1", f"word{n}", 3)] == [f"d{n}#0"]
    stats = store._open().index_stats("text_idx")
    assert stats.num_unindexed_rows < OPTIMIZE_EVERY, "a refresh folded earlier rows into the index"
    store.optimize()
    assert store._open().index_stats("text_idx").num_unindexed_rows == 0
    assert [h.chunk.id for h in store.text_search("c1", "word0", 3)] == ["d0#0"]


def test_delete_doc_and_stale_tail(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    put(store, embedder, make_chunks("d2", "c1", SECURITY))
    assert store.delete_doc("d1", keep_ordinals_below=2) == 2
    assert [c.ordinal for c in store.chunks_for("d1")] == [0, 1]
    assert len(store.chunks_for("d2")) == 3, "other documents are untouched"
    assert store.delete_doc("d1", keep_ordinals_below=5) == 0
    assert store.delete_doc("d1") == 2
    assert store.chunks_for("d1") == []
    assert store.text_search("c1", "economy", 5) == []
    assert store.count("c1") == 3


def test_reads_print_no_library_warnings(
    store: LanceDBStore, embedder: FakeEmbedder, capfd: pytest.CaptureFixture[str]
) -> None:
    """LanceDB's core logs a deprecation warning straight to stderr when a search selects columns
    without its score column. Every read names `_distance` or `_score` itself, so nothing is printed."""
    put(store, embedder, make_chunks("t", "c1", TRAVEL))
    capfd.readouterr()
    assert store.vector_search("c1", embedder.embed_query("hotel night"), 3)
    assert store.text_search("c1", "hotel", 3)
    assert store.has_documents("c1") and store.chunks_for("t") and store.count("c1") == len(TRAVEL)
    captured = capfd.readouterr()
    assert "Deprecation" not in captured.err and "WARN" not in captured.err, captured.err


def test_chunks_for_returns_ordinal_order(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    chunks = make_chunks("d1", "c1", [f"Paragraph {i} text." for i in range(12)])
    put(store, embedder, list(reversed(chunks[:6])))
    put(store, embedder, chunks[6:])
    assert [c.ordinal for c in store.chunks_for("d1")] == list(range(12))


def test_counts_and_delete_collection(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    put(store, embedder, make_chunks("d2", "c2", SECURITY))
    assert (store.count(), store.count("c1"), store.count("c2")) == (7, 4, 3)
    assert store.has_documents("c1") and not store.has_documents("c3")
    assert store.delete_collection("c1") == 4
    assert (store.count(), store.has_documents("c1"), store.has_documents("c2")) == (3, False, True)


def test_collection_isolation_is_a_hard_gate(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    """A search in collection A never returns a chunk of collection B, even when B matches better
    and A has fewer chunks than k."""
    put(store, embedder, make_chunks("a1", "A", ["The cafeteria opens at eight."]))
    put(store, embedder, make_chunks("b1", "B", SECURITY + TRAVEL))
    for query in ["passwords fourteen characters", "hotel capital cities", "laptops lock minutes"]:
        vector_hits = store.vector_search("A", embedder.embed_query(query), 50)
        text_hits = store.text_search("A", query, 50)
        assert {h.chunk.collection_id for h in vector_hits} <= {"A"}
        assert {h.chunk.collection_id for h in text_hits} <= {"A"}
        assert len(vector_hits) == 1
    assert all(h.chunk.collection_id == "B" for h in store.text_search("B", "passwords", 50))
    assert store.text_search("A", "passwords", 50) == []


def test_a_bigger_collection_cannot_crowd_out_a_smaller_one(
    store: LanceDBStore, embedder: FakeEmbedder
) -> None:
    """Filtering happens before ranking: B's many closer matches never push A's chunk out of A's top k."""
    put(store, embedder, make_chunks("a1", "A", ["Passwords are reviewed yearly by the auditors."]))
    put(
        store,
        embedder,
        make_chunks("b1", "B", [f"Passwords must contain fourteen characters, rule {i}." for i in range(40)]),
    )
    query = "passwords must contain fourteen characters"
    assert [h.chunk.id for h in store.vector_search("A", embedder.embed_query(query), 3)] == ["a1#0"]
    assert [h.chunk.id for h in store.text_search("A", query, 3)] == ["a1#0"]


def test_quotes_in_ids_cannot_widen_a_filter(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("d1", "c1", TRAVEL))
    sneaky = "x' OR collection_id = 'c1"
    assert store.vector_search(sneaky, embedder.embed_query("hotel"), 10) == []
    assert store.text_search(sneaky, "hotel", 10) == []
    assert store.delete_doc("d' OR doc_id = 'd1") == 0
    assert store.count(sneaky) == 0 and store.count("c1") == 4


def test_vector_count_and_dimension_checks(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    chunks = make_chunks("d1", "c1", TRAVEL)
    with pytest.raises(ValueError, match="vectors"):
        store.upsert(chunks, embedder.embed([c.text for c in chunks])[:2])
    with pytest.raises(ValueError, match="dimensions"):
        store.upsert(chunks[:1], [[0.1, 0.2]])


def test_dimension_mismatch_with_an_existing_index(settings, embedder: FakeEmbedder) -> None:
    first = LanceDBStore(settings.lancedb_dir, dims=embedder.dims)
    put(first, embedder, make_chunks("d1", "c1", TRAVEL))
    other = LanceDBStore(settings.lancedb_dir, dims=8)
    with pytest.raises(StoreError, match="384-dimension"):
        other.vector_search("c1", [0.1] * 8, 5)
    assert other.count("c1") == 4, "non-vector reads keep working"


def test_data_survives_a_new_store_instance(settings, embedder: FakeEmbedder) -> None:
    first = LanceDBStore(settings.lancedb_dir, dims=embedder.dims)
    put(first, embedder, make_chunks("d1", "c1", TRAVEL))
    second = LanceDBStore(settings.lancedb_dir, dims=embedder.dims)
    assert second.count("c1") == 4
    assert second.text_search("c1", "allowance", 3)[0].chunk.id == "d1#2"
    put(first, embedder, make_chunks("d2", "c1", SECURITY))
    assert second.count("c1") == 7, "a second instance sees writes made through the first"


def test_concurrent_reads_during_writes(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    put(store, embedder, make_chunks("seed", "c1", TRAVEL))
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for n in range(5):
                put(store, embedder, make_chunks(f"w{n}", "c1", SECURITY))
        except BaseException as exc:  # noqa: BLE001 - report any failure from the thread
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(10):
                store.vector_search("c1", embedder.embed_query("hotel"), 5)
                store.text_search("c1", "laptops", 5)
        except BaseException as exc:  # noqa: BLE001 - report any failure from the thread
            errors.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count("c1") == 4 + 5 * 3


# ========================================================================================== SQLite


def make_document(collection_id: str, **overrides) -> Document:
    fields = {
        "id": overrides.pop("id", "doc1"),
        "collection_id": collection_id,
        "title": "Travel Policy",
        "filename": "travel.md",
        "media_type": "text/markdown",
        "size_bytes": 120,
        "content_hash": "abc",
        "uploaded_at": NOW,
    }
    fields.update(overrides)
    return Document(**fields)


def test_meta_creates_wal_database(settings) -> None:
    meta = SQLiteMeta(settings.meta_path)
    assert settings.meta_path.exists()
    with meta._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    SQLiteMeta(settings.meta_path)  # reopening is safe


def test_collections_crud(meta: SQLiteMeta) -> None:
    first = meta.create_collection("  Policies ", "HR and finance")
    second = meta.create_collection("Products")
    assert first.name == "Policies" and len(first.id) == 32
    assert meta.get_collection(first.id) == first
    assert [c.id for c in meta.list_collections()] == [first.id, second.id]
    assert meta.get_collection("missing") is None
    meta.delete_collection(first.id)
    assert [c.id for c in meta.list_collections()] == [second.id]


def test_documents_crud_and_lookups(meta: SQLiteMeta) -> None:
    c = meta.create_collection("C")
    doc = meta.create_document(make_document(c.id, metadata={"source": "upload"}))
    assert meta.get_document("doc1") == doc
    updated = meta.update_document(
        "doc1",
        status=DocumentStatus.ready,
        chunk_count=7,
        pages=3,
        indexed_at=NOW,
        knowledge_version="v20261010.1",
    )
    assert (updated.status, updated.chunk_count, updated.pages, updated.indexed_at) == (
        DocumentStatus.ready,
        7,
        3,
        NOW,
    )
    assert updated.metadata == {"source": "upload"}
    assert meta.find_by_hash(c.id, "abc").id == "doc1"
    assert meta.find_by_hash(c.id, "other") is None
    assert meta.find_by_title(c.id, "  travel POLICY ").id == "doc1"
    assert meta.find_by_title("other-collection", "Travel Policy") is None
    meta.create_document(
        make_document(c.id, id="doc2", title="Second", uploaded_at=datetime(2026, 10, 11, tzinfo=UTC))
    )
    assert [d.id for d in meta.list_documents(c.id)] == ["doc2", "doc1"], "newest first"
    meta.delete_document("doc1")
    assert meta.get_document("doc1") is None


def test_update_document_rejects_bad_fields_and_missing_rows(meta: SQLiteMeta) -> None:
    c = meta.create_collection("C")
    meta.create_document(make_document(c.id))
    with pytest.raises(ValueError, match="nonsense"):
        meta.update_document("doc1", nonsense=1)
    with pytest.raises(ValueError, match="id"):
        meta.update_document("doc1", id="other")
    with pytest.raises(KeyError):
        meta.update_document("missing", chunk_count=1)


def test_jobs(meta: SQLiteMeta) -> None:
    c = meta.create_collection("C")
    job = meta.create_job(c.id, "doc1")
    assert job.status == DocumentStatus.queued and job.finished_at is None
    running = meta.update_job(job.id, DocumentStatus.processing)
    assert running.finished_at is None
    done = meta.update_job(job.id, DocumentStatus.failed, error="The file is empty.")
    assert done.status == DocumentStatus.failed and done.error == "The file is empty." and done.finished_at
    assert [j.id for j in meta.list_jobs(c.id)] == [job.id]
    assert meta.get_job("missing") is None
    with pytest.raises(KeyError):
        meta.update_job("missing", DocumentStatus.ready)


def test_job_records_the_document_it_produced(meta: SQLiteMeta) -> None:
    c = meta.create_collection("C")
    job = meta.create_job(c.id, "")  # a new upload: no document yet
    assert meta.update_job(job.id, DocumentStatus.processing).doc_id == ""
    assert meta.update_job(job.id, DocumentStatus.ready, doc_id="doc9").doc_id == "doc9"
    # A later update without a document id keeps the recorded one.
    assert meta.update_job(job.id, DocumentStatus.failed, "Re-index failed.").doc_id == "doc9"
    assert meta.get_job(job.id).doc_id == "doc9"


def test_answers_and_feedback(meta: SQLiteMeta) -> None:
    c1, c2 = meta.create_collection("C1"), meta.create_collection("C2")
    meta.save_answer('{"id": "a1"}', "a1", c1.id)
    meta.save_answer('{"id": "a2"}', "a2", c2.id)
    assert meta.get_answer("a1") == '{"id": "a1"}'
    assert meta.get_answer("missing") is None
    meta.add_feedback(Feedback(answer_id="a1", rating="up", note="clear", created_at=NOW))
    meta.add_feedback(Feedback(answer_id="a2", rating="down", created_at=NOW))
    assert [f.rating for f in meta.list_feedback()] == ["down", "up"]
    assert [f.note for f in meta.list_feedback(c1.id)] == ["clear"]
    meta.delete_collection(c1.id)
    assert meta.get_answer("a1") is None and meta.list_feedback(c1.id) == []
    assert meta.get_answer("a2") is not None


def test_audit_log(meta: SQLiteMeta) -> None:
    meta.audit("ingest", doc_id="d1", chunks=4, at_time=NOW, path=Path("x"))
    meta.audit("delete", doc_id="d1")
    rows = meta.list_audit()
    assert [r["event"] for r in rows] == ["delete", "ingest"]
    assert rows[1]["chunks"] == 4 and rows[1]["at_time"] == NOW.isoformat()
    assert [r["event"] for r in meta.list_audit(event="ingest")] == ["ingest"]


def test_version_format_increments_and_resets_daily(
    meta: SQLiteMeta, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert meta.current_version() == ""
    monkeypatch.setattr(sqlite_meta, "_now", lambda: datetime(2026, 10, 10, 23, 59, tzinfo=UTC))
    assert meta.bump_version() == "v20261010.1"
    assert meta.bump_version() == "v20261010.2"
    assert meta.current_version() == "v20261010.2"
    monkeypatch.setattr(sqlite_meta, "_now", lambda: datetime(2026, 10, 11, 0, 1, tzinfo=UTC))
    assert meta.bump_version() == "v20261011.1"
    assert re.fullmatch(r"v\d{8}\.\d+", meta.current_version())


def test_concurrent_version_bumps_are_unique(meta: SQLiteMeta) -> None:
    results: list[str] = []
    lock = threading.Lock()

    def bump() -> None:
        for _ in range(10):
            value = meta.bump_version()
            with lock:
                results.append(value)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 40 and len(set(results)) == 40


def test_meta_is_shared_safely_across_threads(meta: SQLiteMeta) -> None:
    c = meta.create_collection("C")
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(10):
                doc_id = f"t{n}-{i}"
                meta.create_document(make_document(c.id, id=doc_id, content_hash=doc_id))
                meta.update_document(doc_id, status=DocumentStatus.ready)
                meta.audit("ingest", doc_id=doc_id)
        except BaseException as exc:  # noqa: BLE001 - report any failure from the thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(meta.list_documents(c.id)) == 40
