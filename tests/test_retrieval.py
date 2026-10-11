"""Retrieval: embedders, rerankers and the hybrid Retriever (fusion, exact terms, candidate pool,
adaptive passage count, neighbour expansion, relevance bands, floor, dedupe, degradation)."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter

import ragkb.retrieval.base as base_module
import ragkb.retrieval.reranker as reranker_module
from ragkb.config import Settings
from ragkb.models import Chunk, Collection, Hit
from ragkb.retrieval.base import (
    Band,
    RankedHit,
    best_relevance,
    classify,
    drop_near_duplicates,
    exact_pattern,
    exact_terms,
    lexical_weights,
    passage_chars,
    relevance_of,
    rrf,
)
from ragkb.retrieval.embedder import EmbedderError, FakeEmbedder, FastEmbedEmbedder, model_prefixes
from ragkb.retrieval.reranker import LocalReranker, NoReranker
from ragkb.retrieval.service import EXACT_MAX_CHUNKS, RetrievalUnavailable, Retriever
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


def test_fused_score_is_rarity_weighted_reciprocal_rank_fusion(
    filled: LanceDBStore, embedder: FakeEmbedder
) -> None:
    retriever = Retriever(filled, embedder)
    query = "reset the monitor to factory settings"
    vector = filled.vector_search("c1", embedder.embed_query(query), 30)
    text = filled.text_search("c1", query, 30)
    weights = lexical_weights([h.lexical_score if h.lexical_score is not None else h.score for h in text])
    expected: dict[str, float] = {}
    for rank, hit in enumerate(vector):
        expected[hit.chunk.id] = expected.get(hit.chunk.id, 0.0) + 1.0 / (60 + rank + 1)
    for rank, (hit, weight) in enumerate(zip(text, weights, strict=True)):
        expected[hit.chunk.id] = expected.get(hit.chunk.id, 0.0) + weight / (60 + rank + 1)
    hits = retriever.search("c1", query, 3)
    assert hits
    for hit in hits:
        assert math.isclose(hit.score, expected[hit.chunk.id])
    assert all(isinstance(h, RankedHit) and h.lexical_score is not None for h in hits if h.lexical_rank)


def test_lexical_weights_follow_bm25_relative_to_the_best() -> None:
    # At least half the best score keeps the full weight; below that the weight falls linearly.
    assert lexical_weights([4.0, 3.0, 2.0, 1.0, 0.04]) == [1.0, 1.0, 1.0, 0.5, 0.02]
    assert lexical_weights([0.0, None, -1.0]) == [1.0, 1.0, 1.0], "no usable scores is plain fusion"
    assert lexical_weights([]) == []


def test_found_by_both_lists_beats_found_by_one(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    hits = Retriever(filled, embedder).search("c1", "battery months interval", 5)
    both = [h for h in hits if h.vector_score is not None and h.lexical_rank is not None]
    single = [h for h in hits if h.vector_score is None or h.lexical_rank is None]
    assert both and both[0].chunk.id == "faq#0"
    assert all(b.score > s.score for b in both[:1] for s in single)


def test_minimum_score_applies_to_vectors_only(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    hits = Retriever(filled, embedder, min_score=0.99).search("c1", "firmware", 5)
    assert [h.chunk.id for h in hits] == ["faq#2"]
    assert hits[0].lexical_rank == 1
    # Found by full text only, it still carries its cosine (below the floor, which it bypasses).
    assert hits[0].vector_score is not None and hits[0].vector_score < 0.99
    assert Retriever(filled, embedder, min_score=0.99).search("c1", "zebra crossing", 5) == []


def test_every_returned_hit_carries_its_cosine(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    query = "Report a lost laptop within one hour"
    hits = Retriever(filled, embedder).search("c1", query, 9)
    assert hits and all(h.vector_score is not None for h in hits)
    by_id = {h.chunk.id: h for h in filled.vector_search("c1", embedder.embed_query(query), 30)}
    for hit in hits:
        if hit.chunk.id in by_id:
            assert math.isclose(hit.vector_score, by_id[hit.chunk.id].vector_score, abs_tol=1e-5)


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


class Reverse:
    """A reranker that reverses the pool and records what it saw."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        self.seen.update(query=query, hits=hits, top_n=top_n)
        return [h.model_copy(update={"rerank_score": float(i)}) for i, h in enumerate(reversed(hits))][:top_n]


def test_reranker_reorders_the_whole_candidate_pool(filled: LanceDBStore, embedder: FakeEmbedder) -> None:
    reverse = Reverse()
    reranked = Retriever(filled, embedder, reranker=reverse).search("c1", "laptop lost hour", 2)
    candidates: list[Hit] = reverse.seen["hits"]  # type: ignore[assignment]
    assert reverse.seen["query"] == "laptop lost hour"
    assert reverse.seen["top_n"] == len(candidates) == 9, "the reranker sees every fused, deduplicated candidate"
    assert [h.chunk.id for h in reranked] == [h.chunk.id for h in reversed(candidates)][:2]
    assert reranked[0].rerank_score == 0.0


def test_candidate_pool_size_comes_from_the_setting(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    index(store, embedder, [chunk("pool", i, f"Overtime rule number {i} for field staff.") for i in range(40)])
    reverse = Reverse()
    Retriever(store, embedder, reranker=reverse, candidates=12).search("c1", "overtime rule", 3)
    assert len(reverse.seen["hits"]) == 12  # type: ignore[arg-type]
    settings = Settings(_env_file=None, CANDIDATES=25, NEIGHBOUR_EXPANSION=False, MIN_SCORE=0.2)
    built = Retriever.from_settings(store, embedder, settings)
    assert (built.candidates, built.neighbour_expansion, built.min_score) == (25, False, 0.2)


# ------------------------------------------------------------------------- exact terms and the haystack


def test_exact_terms_find_identifiers_and_quoted_phrases() -> None:
    assert exact_terms("What does SKU-04417 cost?") == ["SKU-04417"]
    assert exact_terms('Is FH-MNT-01 or fh200 listed under "core hours" on 2026-12-28?') == [
        "core hours", "FH-MNT-01", "fh200", "2026-12-28",
    ]
    # Plain words, plain numbers, short codes and decimals are not identifiers.
    assert exact_terms("How many days of leave in 2022 for a full-time A4 role at 1.5 times?") == []


def test_exact_pattern_matches_whole_tokens_with_any_separator() -> None:
    pattern = exact_pattern("SKU-04417")
    for text in ("| SKU-04417 | Bracket |", "sku04417", "SKU 04417 and more", "(SKU/04417)"):
        assert pattern.search(text), text
    for text in ("SKU-044170", "XSKU-04417", "SKU-0441"):
        assert not pattern.search(text), text
    assert exact_pattern("fh200").search("The FH-200 monitor")
    assert exact_pattern("core hours").search("Core\nhours are 10:00 to 15:00")


HEADER = "| SKU | Product | Description | Unit price (EUR) |\n|---|---|---|---|"
TARGET_SKU = "SKU-04417"


def haystack_rows() -> list[str]:
    """600 price-list rows. The section around SKU-04417 has long descriptions, so its chunk sits far
    from the query in vector space while every chunk shares the common token "SKU"."""
    rows = []
    for i in range(600):
        sku = f"SKU-{4000 + i * 3:05d}"
        if 120 <= i < 140:
            word = "galvanised" if sku == TARGET_SKU else f"model{i}"
            description = (f"{word} bracket for outdoor poles, variant {i}, with long bolts, rubber washers, "
                           f"a printed fitting template and spare clips type{i} size{i % 7}")
        else:
            description = "Steel bracket, pack of ten"
        rows.append(f"| {sku} | Bracket {i} | {description} | {10 + i % 40} |")
    return rows


@pytest.fixture
def haystack(store: LanceDBStore, embedder: FakeEmbedder) -> LanceDBStore:
    """One spreadsheet in 30 chunks of 20 rows, the header repeated in every chunk."""
    rows = haystack_rows()
    chunks = []
    for n in range(30):
        text = HEADER + "\n" + "\n".join(rows[n * 20:(n + 1) * 20])
        c = chunk("sheet", n, text)
        chunks.append(c.model_copy(update={"title": "Price list", "section": "Sheet1",
                                           "embed_text": f"Price list > Sheet1: {text}"}))
    index(store, embedder, chunks)
    return store


def plain_rrf_rank(store: LanceDBStore, embedder: FakeEmbedder, query: str, chunk_id: str) -> int:
    vector = store.vector_search("c1", embedder.embed_query(query), 30)
    text = store.text_search("c1", query, 30)
    fused = rrf([[h.chunk.id for h in vector], [h.chunk.id for h in text]])
    return sorted(fused, key=lambda cid: -fused[cid]).index(chunk_id) + 1


@pytest.mark.parametrize("query", [TARGET_SKU, f"What does {TARGET_SKU} cost?"])
def test_exact_identifier_ranks_first_in_a_haystack_of_near_identical_rows(
    haystack: LanceDBStore, embedder: FakeEmbedder, query: str
) -> None:
    target = "sheet#6"
    assert TARGET_SKU in next(c.text for c in haystack.chunks_for("sheet") if c.id == target)
    lexical = haystack.text_search("c1", query, 30)
    assert lexical[0].chunk.id == target and len(lexical) == 30, "every chunk matches the common token"
    # The report's failure: equal-weight fusion buries the only chunk with the rare token.
    assert plain_rrf_rank(haystack, embedder, query, target) > 6
    hits = Retriever(haystack, embedder).search("c1", query, 6)
    assert hits[0].chunk.id == target
    assert hits[0].exact_match and hits[0].lexical_rank == 1
    assert all(not h.exact_match for h in hits[1:])


def test_rare_word_wins_by_rarity_weighting_without_an_identifier(
    haystack: LanceDBStore, embedder: FakeEmbedder
) -> None:
    query = "galvanised steel bracket pack"
    target = "sheet#6"
    assert plain_rrf_rank(haystack, embedder, query, target) > 1
    hits = Retriever(haystack, embedder).search("c1", query, 6)
    assert hits[0].chunk.id == target and not hits[0].exact_match


def test_a_common_identifier_gets_no_boost(store: LanceDBStore, embedder: FakeEmbedder) -> None:
    common = [chunk("faq", i, f"The FH-200 monitor note {i}: battery, mounting and Wi-Fi.") for i in range(6)]
    index(store, embedder, common)
    hits = Retriever(store, embedder).search("c1", "FH-200 battery", 6)
    assert len(hits) == 6 and EXACT_MAX_CHUNKS < 6
    assert not any(h.exact_match for h in hits)


def test_exact_match_stays_first_after_reranking(haystack: LanceDBStore, embedder: FakeEmbedder) -> None:
    reverse = Reverse()
    hits = Retriever(haystack, embedder, reranker=reverse).search("c1", TARGET_SKU, 6)
    assert hits[0].chunk.id == "sheet#6" and hits[0].exact_match
    assert reverse.seen["hits"][0].chunk.id == "sheet#6"  # type: ignore[index]
    assert hits[0].rerank_score is not None


# ------------------------------------------------------------------------- adaptive count and neighbours


@pytest.fixture
def short_passages(store: LanceDBStore, embedder: FakeEmbedder) -> LanceDBStore:
    index(store, embedder, [chunk(f"note{i}", 0, f"Overtime note {i}: overtime is agreed with the manager in advance.")
                            for i in range(20)])
    return store


def test_adaptive_passage_count_fills_the_budget_with_top_k_as_the_minimum(
    short_passages: LanceDBStore, embedder: FakeEmbedder
) -> None:
    retriever = Retriever(short_passages, embedder, neighbour_expansion=False)
    assert len(retriever.search("c1", "overtime manager", 3)) == 3, "without a budget: exactly top_k"
    roomy = retriever.search("c1", "overtime manager", 3, budget_chars=100_000)
    assert len(roomy) == 20, "a large budget takes every candidate that fits"
    cost = passage_chars(roomy[0].chunk) + 2
    some = retriever.search("c1", "overtime manager", 3, budget_chars=cost * 7)
    assert len(some) == 7
    assert sum(passage_chars(h.chunk) for h in some) + 2 * (len(some) - 1) <= cost * 7
    tight = retriever.search("c1", "overtime manager", 3, budget_chars=10)
    assert len(tight) == 3, "top_k is the minimum; the composer trims what does not fit"
    assert [h.chunk.id for h in some] == [h.chunk.id for h in roomy[:7]], "rank order is kept"


def test_weak_passages_past_top_k_are_left_out(
    short_passages: LanceDBStore, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(base_module.RERANK_BANDS, "calibrated-test", Band(weak_below=-7.0, strong_from=0.0))

    class Calibrated:
        name = "calibrated-test"

        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            scored = [h.model_copy(update={"rerank_score": 5.0 if i < 4 else -9.0}) for i, h in enumerate(hits)]
            return scored[:top_n]

    retriever = Retriever(short_passages, embedder, reranker=Calibrated(), neighbour_expansion=False)
    hits = retriever.search("c1", "overtime manager", 2, budget_chars=100_000)
    assert len(hits) == 4
    assert [h.relevance for h in hits] == ["strong"] * 4
    few = retriever.search("c1", "overtime manager", 6, budget_chars=100_000)
    assert len(few) == 6 and [h.relevance for h in few[4:]] == ["weak", "weak"], "top_k is still the minimum"


TABLE = [
    "Lead times by product.\n| Product | Days |\n|---|---|\n| Monitor | 5 |\n| Power adapter | 3 |",
    "| Product | Days |\n|---|---|\n| Wall mount | 3 |\n| LTE modem kit | 10 |\n| Solar panel kit | 10 |",
    "| Product | Days |\n|---|---|\n| Calibration service | 10 |\n| Extended warranty | 0 |",
    "Orders placed after 14:00 ship the next working day.",
]


@pytest.fixture
def split_table(store: LanceDBStore, embedder: FakeEmbedder) -> LanceDBStore:
    index(store, embedder, [chunk("leadtimes", i, text) for i, text in enumerate(TABLE)])
    index(store, embedder, [chunk("other", 0, "The canteen opens at noon on weekdays.")])
    return store


def test_neighbour_expansion_attaches_the_previous_and_next_chunk(
    split_table: LanceDBStore, embedder: FakeEmbedder
) -> None:
    query = "lead time LTE modem kit"
    # A pool of one: the rest of the table is not among the candidates, as in a large collection where
    # the continuation of a table shares no word with the question.
    hits = Retriever(split_table, embedder, candidates=1).search("c1", query, 1, budget_chars=10_000)
    ids = [h.chunk.id for h in hits]
    top = ids[0]
    assert top == "leadtimes#1"
    attached = [h for h in hits if h.neighbour_of is not None]
    assert {h.chunk.id for h in attached} >= {"leadtimes#0", "leadtimes#2"}
    for hit in attached:
        anchor_position = ids.index(hit.neighbour_of)
        assert ids.index(hit.chunk.id) > anchor_position, "a neighbour comes after its anchor"
        assert hit.score == 0.0 and hit.relevance is None and hit.rerank_score is None
        assert hit.vector_score is not None
    assert ids.index("leadtimes#0") < ids.index("leadtimes#2"), "neighbours in document order"
    # The chunks share the table header, yet neighbours are never dropped as near-duplicates.
    assert len(set(ids)) == len(ids)


def test_neighbour_expansion_only_with_a_budget_that_fits_and_when_enabled(
    split_table: LanceDBStore, embedder: FakeEmbedder
) -> None:
    query = "lead time LTE modem kit"
    assert not any(h.neighbour_of for h in Retriever(split_table, embedder, candidates=1).search("c1", query, 1))
    off = Retriever(split_table, embedder, candidates=1, neighbour_expansion=False)
    assert not any(h.neighbour_of for h in off.search("c1", query, 1, budget_chars=10_000))
    top_only = passage_chars(chunk("leadtimes", 1, TABLE[1])) + 5
    tight = Retriever(split_table, embedder, candidates=1).search("c1", query, 1, budget_chars=top_only)
    assert [h.chunk.id for h in tight] == ["leadtimes#1"]
    # Room for exactly one neighbour: the previous chunk comes first in document order.
    one_more = top_only + passage_chars(chunk("leadtimes", 0, TABLE[0])) + 2
    some = Retriever(split_table, embedder, candidates=1).search("c1", query, 1, budget_chars=one_more)
    assert [h.chunk.id for h in some] == ["leadtimes#1", "leadtimes#0"]


class NoNeighbours:
    """The store without a neighbours method, as a store written before it existed."""

    def __init__(self, store: LanceDBStore):
        self._store = store
        self.chunks_for_calls = 0

    def vector_search(self, *args, **kwargs):
        return self._store.vector_search(*args, **kwargs)

    def text_search(self, *args, **kwargs):
        return self._store.text_search(*args, **kwargs)

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        self.chunks_for_calls += 1
        return self._store.chunks_for(doc_id)


def test_neighbour_expansion_falls_back_to_chunks_for(split_table: LanceDBStore, embedder: FakeEmbedder) -> None:
    proxy = NoNeighbours(split_table)
    retriever = Retriever(proxy, embedder, candidates=1)  # type: ignore[arg-type]
    hits = retriever.search("c1", "lead time LTE modem kit", 1, budget_chars=10_000)
    assert proxy.chunks_for_calls >= 1
    assert {h.chunk.id for h in hits if h.neighbour_of} >= {"leadtimes#0", "leadtimes#2"}


def test_a_failing_neighbour_lookup_never_fails_the_search(
    split_table: LanceDBStore, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(split_table, "neighbours", broken)
    retriever = Retriever(split_table, embedder, candidates=1)
    hits = retriever.search("c1", "lead time LTE modem kit", 1, budget_chars=10_000)
    assert [h.chunk.id for h in hits] == ["leadtimes#1"]


# ------------------------------------------------------------------------- relevance bands


def test_relevance_comes_from_the_calibrated_bands(
    filled: LanceDBStore, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    uncalibrated = Retriever(filled, embedder).search("c1", "passwords fourteen characters", 3)
    assert all(h.relevance is None for h in uncalibrated), "no band for the test embedder"
    monkeypatch.setitem(base_module.COSINE_BANDS, embedder.name, Band(weak_below=0.2, strong_from=0.5))
    hits = Retriever(filled, embedder).search("c1", "passwords fourteen characters", 3)
    for hit in hits:
        expected = "weak" if hit.vector_score < 0.2 else "strong" if hit.vector_score >= 0.5 else "borderline"
        assert hit.relevance == expected


def test_classify_strong_if_any_score_is_strong_weak_only_if_all_are() -> None:
    rerank, cosine = Band(-7.0, 0.0), Band(0.60, 0.66)
    hit = Hit(chunk=chunk("d", 0, "x"), score=0.03, vector_score=0.664, rerank_score=-3.3)
    assert classify(hit, rerank=rerank, cosine=cosine) == "strong", "the cosine is strong"
    hit = Hit(chunk=chunk("d", 0, "x"), score=0.03, vector_score=0.615, rerank_score=-8.7)
    assert classify(hit, rerank=rerank, cosine=cosine) == "borderline", "weak by rerank, borderline by cosine"
    hit = Hit(chunk=chunk("d", 0, "x"), score=0.03, vector_score=0.53, rerank_score=-10.9)
    assert classify(hit, rerank=rerank, cosine=cosine) == "weak"
    assert classify(hit, rerank=None, cosine=None) is None
    assert classify(hit, rerank=None, cosine=None, exact_match=True) == "strong"


def test_relevance_of_plain_hits_and_ranked_hits() -> None:
    plain = Hit(chunk=chunk("d", 0, "x"), score=0.03, rerank_score=-9.0)
    assert relevance_of(plain) == "weak", "a plain hit with a rerank score uses the default reranker's band"
    assert relevance_of(Hit(chunk=chunk("d", 0, "x"), score=0.03, vector_score=0.9)) is None
    ranked = RankedHit(chunk=chunk("d", 0, "x"), score=0.03, rerank_score=-9.0, relevance=None)
    assert relevance_of(ranked) is None, "the retriever's verdict wins, even when it has none"
    neighbour = RankedHit(chunk=chunk("d", 1, "y"), score=0.0, relevance="strong", neighbour_of="d#0")
    assert relevance_of(neighbour) is None
    assert best_relevance([ranked, neighbour, plain]) == "weak"
    assert best_relevance([]) is None


def test_ranked_hits_serialize_as_plain_hits_where_the_type_says_hit() -> None:
    ranked = RankedHit(chunk=chunk("d", 0, "x"), score=0.03, relevance="strong", exact_match=True)
    dumped = TypeAdapter(list[Hit]).dump_python([ranked])
    assert set(dumped[0]) == set(Hit.model_fields)


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
