"""Citation markers, dense renumbering, snippets and the language guess."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ragkb.answer.citations import (
    closest,
    content_tokens,
    densify,
    extract_markers,
    guess_language,
    snippet,
    stem,
)
from ragkb.models import Chunk, Hit


def make_hit(n: int, *, doc: str | None = None, chunk_id: str | None = None, text: str | None = None,
             score: float = 0.03, page: int | None = None, section: str = "") -> Hit:
    doc_id = doc or f"doc{n}"
    return Hit(
        chunk=Chunk(
            id=chunk_id or f"{doc_id}#{n}",
            doc_id=doc_id,
            collection_id="col",
            ordinal=n,
            text=text if text is not None else f"Passage number {n} text.",
            embed_text="",
            title=f"Title {n}",
            section=section,
            page=page,
            created_at=datetime.now(UTC),
        ),
        score=score,
    )


def test_extract_markers_distinct_in_order_of_appearance():
    text = "First [1]. Then [3][1]. Also [2, 4] and [ 5 ]. Not a marker: [a] or [^6]."
    assert extract_markers(text) == [1, 3, 2, 4, 5]
    assert extract_markers("") == []


def test_densify_renumbers_by_first_appearance():
    kept = [make_hit(1), make_hit(2), make_hit(3)]
    text, citations = densify("Alpha [3]. Beta [1]. Gamma [3].", kept)
    assert text == "Alpha [1]. Beta [2]. Gamma [1]."
    assert [c.marker for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == [kept[2].chunk.id, kept[0].chunk.id]


def test_densify_dedupes_by_chunk():
    kept = [make_hit(1, chunk_id="same#0"), make_hit(2), make_hit(3, chunk_id="same#0")]
    text, citations = densify("One [1]. Two [3]. Three [2].", kept)
    assert text == "One [1]. Two [1]. Three [2]."
    assert [c.chunk_id for c in citations] == ["same#0", kept[1].chunk.id]


def test_densify_leaves_unknown_markers_alone():
    kept = [make_hit(1), make_hit(2)]
    text, citations = densify("Known [2]. Unknown [7]. Zero [0]. Year [2023].", kept)
    assert text == "Known [1]. Unknown [7]. Zero [0]. Year [2023]."
    assert len(citations) == 1


def test_densify_rewrites_in_one_pass_so_ten_is_not_corrupted():
    kept = [make_hit(n) for n in range(1, 11)]
    text, citations = densify("Last [10] then first [1] then second [2].", kept)
    assert text == "Last [1] then first [2] then second [3]."
    assert [c.chunk_id for c in citations] == [kept[9].chunk.id, kept[0].chunk.id, kept[1].chunk.id]


def test_densify_does_not_chain_swaps():
    kept = [make_hit(1), make_hit(2)]
    text, _ = densify("B [2] A [1]", kept)
    assert text == "B [1] A [2]"


def test_densify_groups_and_runs():
    kept = [make_hit(1), make_hit(2), make_hit(3)]
    text, citations = densify("Grouped [3, 1]. Repeated [2][2]. Mixed [2][9].", kept)
    assert text == "Grouped [1][2]. Repeated [3]. Mixed [3][9]."
    assert [c.marker for c in citations] == [1, 2, 3]


def test_densify_citation_fields():
    long_text = "Employees receive twenty-five days of paid leave. " * 10
    hit = make_hit(1, text=long_text, page=4, section="Leave > Annual", score=0.0327)
    _, citations = densify("Answer [1].", [hit])
    citation = citations[0]
    assert citation.marker == 1
    assert citation.doc_id == hit.chunk.doc_id and citation.title == "Title 1"
    assert citation.page == 4 and citation.section == "Leave > Annual"
    assert citation.score == pytest.approx(0.0327)
    assert len(citation.snippet) <= 240 and citation.snippet.endswith("...")


def test_densify_citations_carry_every_score_and_the_page_range():
    hit = make_hit(1, score=0.031).model_copy(update={"vector_score": 0.71, "rerank_score": 2.5})
    hit = hit.model_copy(update={"chunk": hit.chunk.model_copy(update={"page": 5, "page_start": 5, "page_end": 6})})
    _, citations = densify("Answer [1].", [hit])
    citation = citations[0]
    assert (citation.score, citation.vector_score, citation.rerank_score) == (0.031, 0.71, 2.5)
    assert (citation.page, citation.page_end) == (5, 6)
    one_page = make_hit(2, page=7).model_copy(update={"chunk": make_hit(2, page=7).chunk.model_copy(
        update={"page_start": 7, "page_end": 7})})
    _, citations = densify("Answer [1].", [one_page])
    assert (citations[0].page, citations[0].page_end) == (7, None)
    no_page = make_hit(3).model_copy(update={"chunk": make_hit(3).chunk.model_copy(
        update={"page": None, "page_start": 2, "page_end": 3})})
    _, citations = densify("Answer [1].", [no_page])
    assert (citations[0].page, citations[0].page_end) == (2, 3)


def test_closest_numbers_the_best_passages_densely():
    hits = [make_hit(n, text=f"Closest passage {n}.", score=0.03 - n / 1000) for n in range(1, 6)]
    hits.insert(1, hits[0])
    cited = closest(hits)
    assert [c.marker for c in cited] == [1, 2, 3]
    assert [c.chunk_id for c in cited] == [hits[0].chunk.id, hits[2].chunk.id, hits[3].chunk.id]
    assert cited[0].snippet == "Closest passage 1." and cited[0].score == pytest.approx(0.029)
    assert closest(hits, limit=1)[0].marker == 1
    assert closest([]) == []


def test_densify_with_no_markers():
    text, citations = densify("No markers here.", [make_hit(1)])
    assert text == "No markers here." and citations == []


def test_snippet():
    assert snippet("Short   text\nwith  breaks") == "Short text with breaks"
    long = "word " * 100
    cut = snippet(long)
    assert len(cut) <= 240 and cut.endswith("...") and not cut[:-3].endswith(" ")
    assert snippet("abcdefghij" * 30, limit=50) == ("abcdefghij" * 5)[:47] + "..."


@pytest.mark.parametrize(
    "question",
    [
        "How many vacation days do new employees get?",
        "What is the policy for working from home?",
        "Can I carry over unused leave to next year?",
        "Who approves expense reports above the limit?",
    ],
)
def test_guess_language_english(question: str):
    assert guess_language(question) == "en"


@pytest.mark.parametrize(
    "question",
    [
        "¿Cuántos días de vacaciones tienen los empleados nuevos?",
        "¿Cuál es la política de trabajo desde casa?",
        "¿Puedo pasar los días de vacaciones no usados al año siguiente?",
        "Quién aprueba los gastos que superan el límite",
    ],
)
def test_guess_language_spanish(question: str):
    assert guess_language(question) == "es"


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Quantos dias de férias têm os novos funcionários?", "pt"),
        ("Combien de jours de congé pour les nouveaux employés ?", "fr"),
        ("Wie viele Urlaubstage bekommen neue Mitarbeiter?", "de"),
        ("Quanti giorni di ferie hanno i nuovi dipendenti?", "it"),
    ],
)
def test_guess_language_other(question: str, language: str):
    assert guess_language(question) == language


@pytest.mark.parametrize("text", ["", "PTO?", "12345", "Kubernetes ingress TLS"])
def test_guess_language_unsure_is_none(text: str):
    assert guess_language(text) is None


def test_content_tokens_and_stem():
    assert stem("policies") == "policy" and stem("days") == "day" and stem("process") == "process"
    assert content_tokens("How many vacation days do new employees get?", "en") == [
        "vacation", "day", "new", "employee",
    ]
    assert content_tokens("¿Cuántos días de vacaciones?", "es") == ["dia", "vacacione"]
