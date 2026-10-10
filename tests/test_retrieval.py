"""Retrieval: embedders, rerankers and the hybrid Retriever (fusion, floor, dedupe, degradation)."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import pytest

import ragkb.retrieval.reranker as reranker_module
from ragkb.models import Chunk, Collection, Hit
from ragkb.retrieval.base import drop_near_duplicates, rrf
from ragkb.retrieval.embedder import EmbedderError, FakeEmbedder, FastEmbedEmbedder, model_prefixes
from ragkb.retrieval.reranker import LocalReranker, NoReranker
from ragkb.retrieval.service import RetrievalUnavailable, Retriever
from ragkb.store.lancedb_store import LanceDBStore

NOW = datetime(2026, 10, 10, tzinfo=UTC)


def chunk(doc_id: str, ordinal: int, text: str, collection_id: str = "c1") -> Chunk:
    return Chunk(
        id=f"{doc_id}#{ordinal}",
        doc_id=doc_id,
        collection_id=collection_id,
        ordinal=ordinal,
        text=text,
        embed_text=text,
        title=doc_id,
        created_at=NOW,
    )


def index(store: LanceDBStore, embedder: FakeEmbedder, chunks: list[Chunk]) -> None:
    store.upsert(chunks, embedder.embed([c.embed_text for c in chunks]))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ----------------------------------------------------------------------------------------- embedders


def test_fake_embedder_is_deterministic_normalized_and_similarity_aware() -> None:
    embedder = FakeEmbedder(dims=64)
    assert embedder.dims == 64 and embedder.name == "fake-hash-64"
    a, b = embedder.embed(["Hotel caps per night", "Hotel caps per night"])
    assert a == b == FakeEmbedder(dims=64).embed_query("Hotel caps per night")
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    query = embedder.embed_query("what is the hotel cap per night")
    related = embedder.embed_query("hotels are capped at 210 euros per night")
    unrelated = embedder.embed_query("laptops lock after five minutes")
    assert cosine(query, related) > cosine(query, unrelated)
    empty = embedder.embed_query("")
    assert math.isclose(sum(x * x for x in empty), 1.0)


def test_unknown_embedding_model_fails_with_a_plain_message() -> None:
    with pytest.raises(EmbedderError) as caught:
        FastEmbedEmbedder("intfloat/multilingual-e5-small")
    message = str(caught.value)
    assert "intfloat/multilingual-e5-small is not supported" in message and "RAGKB_EMBED_MODEL" in message
    assert "BAAI/bge-small-en-v1.5" in message
    assert "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" in message
    assert "Traceback" not in message


def test_the_documented_models_are_known_without_loading_them() -> None:
    # dims come from fastembed's registry; the model itself only loads on the first embedding.
    assert FastEmbedEmbedder("BAAI/bge-small-en-v1.5").dims == 384
    multilingual = FastEmbedEmbedder("sentence-transformers/paraphrase-multilingual-minilm-l12-v2")
    assert multilingual.dims == 384 and multilingual._model is None


def test_model_prefixes() -> None:
    assert model_prefixes("BAAI/bge-small-en-v1.5")[0].startswith("Represent this sentence")
    assert model_prefixes("BAAI/bge-small-en-v1.5")[1] == ""
    assert model_prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert model_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


# ----------------------------------------------------------------------------------------- rerankers


def hits_of(texts: list[str]) -> list[Hit]:
    return [Hit(chunk=chunk("d", i, t), score=1.0 / (i + 1)) for i, t in enumerate(texts)]


def test_no_reranker_keeps_order() -> None:
    hits = hits_of(["a", "b", "c"])
    assert NoReranker().rerank("q", hits, 2) == hits[:2]


class FakeCrossEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, documents):
        return [float(len(d)) for d in documents]  # longer text wins


def test_local_reranker_scores_and_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_module, "TextCrossEncoder", FakeCrossEncoder)
    reranked = LocalReranker("fake-model").rerank("q", hits_of(["aa", "a", "aaaa", "aaa"]), 3)
    assert [h.chunk.text for h in reranked] == ["aaaa", "aaa", "aa"]
    assert [h.rerank_score for h in reranked] == [4.0, 3.0, 2.0]
    assert reranked[0].score == 1.0 / 3, "the fused score is kept"


def test_local_reranker_falls_back_to_input_order_and_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(model_name: str):
        raise OSError("model files are not available offline")

    monkeypatch.setattr(reranker_module, "TextCrossEncoder", broken)
    reranker = LocalReranker("missing-model")
    hits = hits_of(["a", "b", "c"])
    with caplog.at_level(logging.WARNING):
        assert reranker.rerank("q", hits, 2) == hits[:2]
        assert reranker.rerank("q", hits, 3) == hits
    assert len([r for r in caplog.records if "reranker" in r.getMessage()]) == 1
    assert reranker.rerank("q", [], 3) == []


# ----------------------------------------------------------------------------------------- retriever


DOCS = {
    "travel": [
        "Economy class is required for flights shorter than six hours.",
        "Hotel stays in capital cities are capped at 210 euros per night.",
        "The daily allowance for international travel is 70 euros.",
    ],
    "security": [
        "Passwords must contain at least fourteen characters.",
        "Laptops lock automatically after five minutes without activity.",
        "Report a lost laptop to the service desk within one hour.",
    ],
    "faq": [
        "The battery lasts about eighteen months on the standard interval.",
        "Hold the button for ten seconds to reset the monitor to factory settings.",
        "Firmware updates arrive over the air every night.",
    ],
}


@pytest.fixture
def filled(store: LanceDBStore, embedder: FakeEmbedder) -> LanceDBStore:
    for doc_id, texts in DOCS.items():
        index(store, embedder, [chunk(doc_id, i, t) for i, t in enumerate(texts)])
    return store


def test_search_returns_the_best_passage_with_scores(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    hits = Retriever(filled, embedder).search("c1", "How many characters must passwords contain?", 3)
    assert hits[0].chunk.id == "security#0"
    top = hits[0]
    assert top.vector_score is not None and top.lexical_rank == 1 and top.rerank_score is None
    assert len(hits) <= 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_fused_score_is_reciprocal_rank_fusion(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    retriever = Retriever(filled, embedder)
    query = "reset the monitor to factory settings"
    vector = filled.vector_search("c1", embedder.embed_query(query), 12)
    text = filled.text_search("c1", query, 12)
    expected = rrf([[h.chunk.id for h in vector], [h.chunk.id for h in text]], k=60)
    for hit in retriever.search("c1", query, 3):
        assert math.isclose(hit.score, expected[hit.chunk.id])


def test_found_by_both_lists_beats_found_by_one(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    hits = Retriever(filled, embedder).search("c1", "battery months interval", 5)
    both = [h for h in hits if h.vector_score is not None and h.lexical_rank is not None]
    single = [h for h in hits if h.vector_score is None or h.lexical_rank is None]
    assert both and both[0].chunk.id == "faq#0"
    assert all(b.score > s.score for b in both[:1] for s in single)


def test_minimum_score_applies_to_vectors_only(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    hits = Retriever(filled, embedder, min_score=0.99).search("c1", "firmware", 5)
    assert [h.chunk.id for h in hits] == ["faq#2"]
    assert hits[0].vector_score is None and hits[0].lexical_rank == 1
    assert Retriever(filled, embedder, min_score=0.99).search("c1", "zebra crossing", 5) == []


def test_near_duplicates_are_dropped(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    same = "Overtime is paid at one and a half times the hourly rate."
    index(
        store,
        embedder,
        [chunk("a", 0, same), chunk("b", 0, "  overtime is PAID at one and a half times the hourly rate. ")],
    )
    index(
        store,
        embedder,
        [
            chunk("c", 0, "Overtime needs approval in advance."),
            # The chunker's overlap: the next chunk starts with the end of the previous one.
            chunk("c", 1, "Overtime needs approval in advance. Overtime rules apply to field staff."),
        ],
    )
    hits = Retriever(store, embedder).search("c1", "overtime hourly rate", 6)
    ids = [h.chunk.id for h in hits]
    assert len([i for i in ids if i in ("a#0", "b#0")]) == 1, "same text is kept once"
    assert len([i for i in ids if i.startswith("c#")]) == 1, "an overlapping neighbour is kept once"


def test_adjacent_neighbour_with_its_own_text_is_kept(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    index(
        store,
        embedder,
        [
            chunk("rma", 0, "Returns need an RMA number. The RMA number is valid for returns of any product."),
            chunk("rma", 1, "Steps: ship the return within 14 days, then the RMA number expires."),
        ],
    )
    ids = [h.chunk.id for h in Retriever(store, embedder).search("c1", "RMA number return", 6)]
    assert ids.count("rma#0") == 1 and ids.count("rma#1") == 1, "a distinct neighbour is a different section"


PROSE = (
    "Expense claims are filed in the finance portal within thirty days of the purchase. Receipts are "
    "required for every amount above twenty euros, and claims are paid with the next monthly salary."
)


def dedupe_ids(chunks: list[Chunk]) -> list[str]:
    hits = [Hit(chunk=c, score=1.0 / (i + 1)) for i, c in enumerate(chunks)]
    return [h.chunk.id for h in drop_near_duplicates(hits)]


def test_drop_near_duplicates_rules() -> None:
    tail = PROSE[-150:]
    following = f"{tail} Late claims need a manager's approval and a short written reason."
    # Identical normalized text, in any document, is kept once.
    assert dedupe_ids([chunk("a", 0, PROSE), chunk("b", 4, "  " + PROSE.upper() + " ")]) == ["a#0"]
    # An adjacent chunk that repeats the other's last 120 characters is dropped, in either rank order.
    assert dedupe_ids([chunk("d", 0, PROSE), chunk("d", 1, following)]) == ["d#0"]
    assert dedupe_ids([chunk("d", 1, following), chunk("d", 0, PROSE)]) == ["d#1"]
    # An adjacent chunk with its own text is kept, and so is an overlapping chunk that is not adjacent.
    other = "Mileage for a private car is reimbursed per kilometre, with parking and tolls on receipts."
    assert dedupe_ids([chunk("d", 0, PROSE), chunk("d", 1, other)]) == ["d#0", "d#1"]
    assert dedupe_ids([chunk("d", 0, PROSE), chunk("d", 2, following)]) == ["d#0", "d#2"]
    # Another document never counts as a neighbour.
    assert dedupe_ids([chunk("d", 0, PROSE), chunk("e", 1, following)]) == ["d#0", "e#1"]


def test_reranker_reorders_the_final_list(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    seen: dict[str, object] = {}

    class Reverse:
        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            seen.update(query=query, hits=hits, top_n=top_n)
            return [h.model_copy(update={"rerank_score": float(i)}) for i, h in enumerate(reversed(hits))][
                :top_n
            ]

    reranked = Retriever(filled, embedder, reranker=Reverse()).search("c1", "laptop lost hour", 2)
    candidates: list[Hit] = seen["hits"]  # type: ignore[assignment]
    assert seen["query"] == "laptop lost hour" and seen["top_n"] == 2
    assert 2 < len(candidates) <= 6, "the reranker sees the fused, deduplicated pool of top_k * 3"
    assert [h.chunk.id for h in reranked] == [h.chunk.id for h in reversed(candidates)][:2]
    assert reranked[0].rerank_score == 0.0


def test_text_search_failure_degrades_to_vectors(
    filled: LanceDBStore,
    embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*args, **kwargs):
        raise ValueError("Cannot perform full text search unless an INVERTED index has been created")

    monkeypatch.setattr(filled, "text_search", broken)
    with caplog.at_level(logging.WARNING):
        hits = Retriever(filled, embedder).search("c1", "passwords fourteen characters", 3)
    assert hits and hits[0].chunk.id == "security#0"
    assert all(h.lexical_rank is None for h in hits)
    assert "vector search only" in caplog.text


def test_store_failure_raises_retrieval_unavailable(
    filled: LanceDBStore, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(filled, "vector_search", broken)
    with pytest.raises(RetrievalUnavailable, match="Search is unavailable"):
        Retriever(filled, embedder).search("c1", "passwords", 3)


def test_embedder_failure_raises_retrieval_unavailable(filled: LanceDBStore) -> None:
    class Broken(FakeEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("model missing")

    with pytest.raises(RetrievalUnavailable):
        Retriever(filled, Broken()).search("c1", "passwords", 3)


def test_empty_query_or_zero_k_returns_nothing(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    retriever = Retriever(filled, embedder)
    assert retriever.search("c1", "   ", 5) == []
    assert retriever.search("c1", "passwords", 0) == []


def test_unknown_or_empty_collection_returns_nothing(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    assert Retriever(filled, embedder).search("nope", "passwords", 5) == []


def test_retriever_respects_collection_isolation(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    index(store, embedder, [chunk("mine", 0, "Our canteen serves lunch from noon.", collection_id="A")])
    index(
        store,
        embedder,
        [chunk("theirs", i, f"Passwords rule {i}: use fourteen characters.", "B") for i in range(8)],
    )
    hits = Retriever(store, embedder).search("A", "passwords fourteen characters", 6)
    assert [h.chunk.collection_id for h in hits] == ["A"]


@pytest.mark.parametrize(
    ("question", "title", "needle"),
    [
        ("How many characters must a password have?", "IT Security Policy", "at least 14 characters"),
        ("What is the per diem for international travel?", "Travel and Expense Policy", "EUR 70 per day"),
        ("What is the restocking fee for a returned unit?", "Customer Return (RMA) Procedure", "15 percent"),
        ("How do I reset the FH-200 to factory settings?", "FH-200 Air Quality Monitor", "10 seconds"),
        ("How much is the pension match?", "Employee Benefits Overview 2026", "6 percent"),
    ],
)
def test_sample_corpus_questions_find_their_passage(
    indexed_corpus: Collection,
    store: LanceDBStore,
    embedder: FakeEmbedder,
    question: str,
    title: str,
    needle: str,
) -> None:
    hits = Retriever(store, embedder).search(indexed_corpus.id, question, 3)
    assert any(h.chunk.title.startswith(title) and needle in h.chunk.text for h in hits), [
        (h.chunk.title, h.chunk.section) for h in hits
    ]
