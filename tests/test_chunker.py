"""Chunker invariants: hard ceilings, size bound, whole short blocks, table headers, overlap, pages,
metadata, determinism."""

from __future__ import annotations

import base64
import math
import random
import re
from itertools import pairwise
from pathlib import Path

import pytest

from ragkb.ingest.chunker import BLOCK_SEPARATOR, ChunkDraft, _sentence_spans, chunk_blocks, estimate_tokens
from ragkb.ingest.parsers import SUPPORTED_SUFFIXES, parse
from ragkb.models import Block
from ragkb.retrieval.embedder import FakeEmbedder

FULL_STOP = chr(0x3002)  # the ideographic full stop
COMMA = chr(0x3001)  # the ideographic comma


def japanese(n_chars: int, *, stop: bool = True) -> str:
    """Japanese-like text: kana and kanji in sentences of 20 to 40 characters, each ending with the
    ideographic full stop (or the ideographic comma when ``stop`` is false), with no spaces."""
    rng = random.Random(3)
    out: list[str] = []
    size = 0
    while size < n_chars:
        length = rng.randint(20, 40)
        sentence = "".join(chr(rng.choice([0x3042, 0x3044, 0x304B, 0x306E, 0x4E00 + rng.randint(0, 3000)]))
                           for _ in range(length))
        out.append(sentence + (FULL_STOP if stop else COMMA))
        size += length + 1
    return "".join(out)


def strict_counter(text: str) -> int:
    """Stands in for a sub-word tokenizer on text without spaces: one token per three characters."""
    return math.ceil(len(text) / 3)


def squeeze(text: str) -> str:
    return "".join(text.split())


def assert_within_ceilings(chunks: list[ChunkDraft], count, max_tokens: int, max_chars: int) -> None:
    for chunk in chunks:
        assert len(chunk.text) <= max_chars, len(chunk.text)
        assert count(chunk.embed_text) <= max_tokens, count(chunk.embed_text)


def sentence(i: int) -> str:
    return f"Sentence number {i} explains one more rule about expense claims and receipts."


def long_paragraph(n: int, start: int = 0) -> str:
    return " ".join(sentence(i) for i in range(start, start + n))


def doc_text(blocks: list[Block]) -> str:
    return BLOCK_SEPARATOR.join(b.text.strip() for b in blocks)


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three four five six seven eight nine ten") == 13
    assert estimate_tokens("word") == 2
    assert estimate_tokens(FULL_STOP * 3) == 3, "one per CJK character"


def test_invalid_limits_raise() -> None:
    with pytest.raises(ValueError):
        chunk_blocks([Block(text="x")], title="t", target_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        chunk_blocks([Block(text="x")], title="t", target_tokens=10, overlap_tokens=0, max_chars=0)


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_blocks([], title="t", target_tokens=100, overlap_tokens=10) == []


def test_short_blocks_pack_into_one_chunk_with_metadata() -> None:
    blocks = [
        Block(text="First paragraph.", heading_path=["Leave", "Annual"], page=3),
        Block(text="Second paragraph.", heading_path=["Leave", "Annual"], page=3),
    ]
    [chunk] = chunk_blocks(blocks, title="Handbook", target_tokens=100, overlap_tokens=20)
    assert chunk.text == "First paragraph.\n\nSecond paragraph."
    assert chunk.section == "Leave > Annual"
    assert (chunk.page, chunk.page_start, chunk.page_end) == (3, 3, 3)
    assert chunk.embed_text == "Handbook > Leave > Annual: First paragraph.\n\nSecond paragraph."
    assert (chunk.char_start, chunk.char_end) == (0, len(chunk.text))
    assert chunk.token_estimate == estimate_tokens(chunk.text)


def test_token_counts_come_from_the_given_counter() -> None:
    [chunk] = chunk_blocks(
        [Block(text="Three words here.")], title="T", target_tokens=100, overlap_tokens=0, count_tokens=len
    )
    assert chunk.token_estimate == len("Three words here.")
    embedder = FakeEmbedder()
    [chunk] = chunk_blocks(
        [Block(text=FULL_STOP.join(["ab cd"] * 3))],
        title="T",
        target_tokens=100,
        overlap_tokens=0,
        count_tokens=embedder.count_tokens,
    )
    assert chunk.token_estimate == embedder.count_tokens(chunk.text) == 6 + 2


def test_embed_text_without_section() -> None:
    [chunk] = chunk_blocks([Block(text="Body.")], title="Doc", target_tokens=50, overlap_tokens=0)
    assert chunk.section == ""
    assert chunk.embed_text == "Doc: Body."


@pytest.mark.parametrize("target", [40, 120, 350])
def test_no_chunk_exceeds_one_and_a_half_targets(target: int, sample_corpus_dir: Path) -> None:
    blocks = [
        Block(text=long_paragraph(60), heading_path=["A"]),
        Block(
            text="\n".join(f"- item {i} " + "word " * 20 for i in range(30)), heading_path=["A"], kind="list"
        ),
        Block(
            text="\n".join(["Col | Other"] + [f"{i} | value {i} " + "x " * 10 for i in range(60)]),
            kind="table",
        ),
        Block(text="w" * 10 + " " + "word " * 1000, heading_path=["B"]),  # one giant sentence
    ]
    chunks = chunk_blocks(blocks, title="T", target_tokens=target, overlap_tokens=target // 5)
    assert chunks
    assert max(c.token_estimate for c in chunks) <= 1.5 * target
    for path in sample_corpus_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            parsed = parse(path)
            for c in chunk_blocks(
                parsed.blocks, title=parsed.title, target_tokens=target, overlap_tokens=target // 5
            ):
                assert c.token_estimate <= 1.5 * target, (path.name, c.section)


def test_blocks_shorter_than_the_target_are_never_split(sample_corpus_dir: Path) -> None:
    target = 120
    for path in sample_corpus_dir.iterdir():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        parsed = parse(path)
        chunks = chunk_blocks(parsed.blocks, title=parsed.title, target_tokens=target, overlap_tokens=20)
        for block in parsed.blocks:
            if estimate_tokens(block.text) <= target:
                assert any(block.text.strip() in c.text for c in chunks), (path.name, block.text[:60])


def test_a_block_that_fits_moves_whole_to_the_next_chunk() -> None:
    """A block under the target that does not fit the current chunk starts the next chunk intact,
    instead of being split to fill the current one."""
    first = long_paragraph(4)
    second = long_paragraph(7, start=100)
    target = 150
    assert estimate_tokens(second) <= target < estimate_tokens(first) + estimate_tokens(second)
    chunks = chunk_blocks(
        [Block(text=first), Block(text=second)], title="T", target_tokens=target, overlap_tokens=0
    )
    assert [c.text for c in chunks] == [first, second]


def test_long_paragraph_splits_on_sentence_boundaries_with_overlap() -> None:
    blocks = [Block(text=long_paragraph(40), heading_path=["Claims"])]
    chunks = chunk_blocks(blocks, title="Policy", target_tokens=100, overlap_tokens=30)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.endswith("receipts.")
        assert chunk.text.startswith("Sentence number")
        assert chunk.section == "Claims"
    for previous, current in pairwise(chunks):
        last_sentence = re.findall(r"Sentence number \d+ [^.]*\.", previous.text)[-1]
        assert current.text.startswith(last_sentence), "the next chunk starts with the previous chunk's tail"
        assert current.char_start < previous.char_end
    numbers = sorted({int(n) for c in chunks for n in re.findall(r"number (\d+)", c.text)})
    assert numbers == list(range(40)), "every sentence appears"


def test_overlap_respects_its_budget() -> None:
    chunks = chunk_blocks([Block(text=long_paragraph(40))], title="T", target_tokens=100, overlap_tokens=20)
    per_sentence = estimate_tokens(sentence(10))
    for previous, current in pairwise(chunks):
        shared = [s for s in re.findall(r"Sentence number \d+", current.text) if s in previous.text]
        assert 1 <= len(shared) <= max(1, 20 // per_sentence)


def test_no_overlap_across_sections() -> None:
    blocks = [
        Block(text=long_paragraph(12), heading_path=["One"]),
        Block(text=long_paragraph(12, start=100), heading_path=["Two"]),
    ]
    chunks = chunk_blocks(blocks, title="T", target_tokens=200, overlap_tokens=40)
    two = [c for c in chunks if c.section == "Two"]
    assert two and all("number 1 " not in c.text and "number 11 " not in c.text for c in two)


def test_table_split_repeats_the_header_and_keeps_every_row_once() -> None:
    header = "Record type | Retention | Owner"
    rows = [f"Record {i} | {i} years after closing the file | Team {i % 4}" for i in range(50)]
    blocks = [Block(text="\n".join([header, *rows]), kind="table", heading_path=["Periods"])]
    chunks = chunk_blocks(blocks, title="Retention", target_tokens=80, overlap_tokens=20)
    assert len(chunks) > 1
    seen: list[str] = []
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert lines[0] == header
        seen.extend(lines[1:])
        assert chunk.section == "Periods"
    assert seen == rows


def test_small_table_stays_whole() -> None:
    table = "A | B\n1 | 2\n3 | 4"
    blocks = [Block(text="Intro sentence."), Block(text=table, kind="table")]
    [chunk] = chunk_blocks(blocks, title="T", target_tokens=100, overlap_tokens=10)
    assert table in chunk.text


def test_heading_travels_with_the_following_block() -> None:
    blocks = [
        Block(text=long_paragraph(6), heading_path=["Intro"]),
        Block(text="Sick leave", heading_path=["Sick leave"], kind="heading"),
        Block(text=long_paragraph(6, start=50), heading_path=["Sick leave"]),
    ]
    chunks = chunk_blocks(blocks, title="T", target_tokens=90, overlap_tokens=0)
    for chunk in chunks:
        assert not chunk.text.endswith("Sick leave")
    assert any(c.text.startswith("Sick leave\n\nSentence number 50") for c in chunks)


def test_sibling_sections_pack_and_take_the_parent_label() -> None:
    blocks = [
        Block(text="Annual leave", heading_path=["Leave", "Annual leave"], kind="heading"),
        Block(text="You get 27 days.", heading_path=["Leave", "Annual leave"]),
        Block(text="Sick leave", heading_path=["Leave", "Sick leave"], kind="heading"),
        Block(text="Self-certify up to 3 days.", heading_path=["Leave", "Sick leave"]),
    ]
    [chunk] = chunk_blocks(blocks, title="Handbook", target_tokens=200, overlap_tokens=20)
    assert chunk.section == "Leave"


def test_new_top_level_topic_starts_a_new_chunk_once_the_chunk_has_substance() -> None:
    blocks = [
        Block(text=long_paragraph(5), heading_path=["Leave"]),
        Block(text="Remote work", heading_path=["Remote work"], kind="heading"),
        Block(text=long_paragraph(2, start=20), heading_path=["Remote work"]),
    ]
    chunks = chunk_blocks(blocks, title="Handbook", target_tokens=200, overlap_tokens=20)
    assert [c.section for c in chunks] == ["Leave", "Remote work"]


def test_tiny_unrelated_sections_share_a_chunk_with_a_joined_label() -> None:
    blocks = [
        Block(text="Intro without heading."),
        Block(text="Devices", heading_path=["Devices"], kind="heading"),
        Block(text="Lock your screen.", heading_path=["Devices"]),
        Block(text="Networks", heading_path=["Networks"], kind="heading"),
        Block(text="Use the VPN.", heading_path=["Networks"]),
    ]
    [chunk] = chunk_blocks(blocks, title="Security", target_tokens=200, overlap_tokens=20)
    assert chunk.section == "Devices, Networks"
    assert chunk.text.startswith("Intro without heading.")


def test_page_is_where_the_chunk_starts() -> None:
    blocks = [Block(text=long_paragraph(3), page=1), Block(text=long_paragraph(3, start=10), page=2)]
    chunks = chunk_blocks(blocks, title="T", target_tokens=60, overlap_tokens=0)
    assert [c.page for c in chunks] == [1, 2]


def test_chunks_never_span_pages() -> None:
    """Short pages are not packed together, so every chunk's page is exact; overlap never crosses a page."""
    blocks = [
        Block(text="Fact one is on the first page.", page=1),
        Block(text="Fact two is also on the first page.", page=1),
        Block(text="Fact three is on page two.", page=2),
        Block(text=long_paragraph(12), page=3),
        Block(text=long_paragraph(12, start=50), page=4),
    ]
    chunks = chunk_blocks(blocks, title="T", target_tokens=60, overlap_tokens=20)
    for chunk in chunks:
        assert chunk.page == chunk.page_start == chunk.page_end
    assert chunks[0].text == "Fact one is on the first page.\n\nFact two is also on the first page."
    assert (chunks[1].text, chunks[1].page) == ("Fact three is on page two.", 2)
    page_four = [c for c in chunks if c.page == 4]
    assert page_four and all("number 11 " not in c.text for c in page_four), "no overlap from page 3"
    page_three = " ".join(c.text for c in chunks if c.page == 3)
    assert not re.search(r"number (5\d|6[01]) ", page_three), "page 4 text stays on page 4"


def test_a_heading_ending_a_page_travels_with_the_next_page() -> None:
    blocks = [
        Block(text="Intro text on page one.", heading_path=[], page=1),
        Block(text="Refunds", heading_path=["Refunds"], page=1, kind="heading"),
        Block(text="Refunds are paid within 14 days.", heading_path=["Refunds"], page=2),
    ]
    first, second = chunk_blocks(blocks, title="T", target_tokens=200, overlap_tokens=0)
    assert (first.text, first.page) == ("Intro text on page one.", 1)
    assert second.text == "Refunds\n\nRefunds are paid within 14 days."
    assert (second.page, second.page_start, second.page_end, second.section) == (2, 2, 2, "Refunds")


def test_formats_without_pages_have_no_page_span() -> None:
    [chunk] = chunk_blocks([Block(text="No pages here.")], title="T", target_tokens=50, overlap_tokens=0)
    assert (chunk.page, chunk.page_start, chunk.page_end) == (None, None, None)


# ------------------------------------------------------------------------------------ hard ceilings


def test_a_200_kb_line_without_spaces_is_cut_into_windows() -> None:
    line = "x" * 200_000
    chunks = chunk_blocks([Block(text=line)], title="T", target_tokens=350, overlap_tokens=60, max_chars=4000)
    assert len(chunks) >= 50
    assert all(len(c.text) <= 4000 for c in chunks)
    assert squeeze("".join(c.text for c in chunks)) == line
    strict = chunk_blocks(
        [Block(text=line)], title="T", target_tokens=350, overlap_tokens=60, count_tokens=strict_counter, max_tokens=510
    )
    assert_within_ceilings(strict, strict_counter, 510, 4000)
    assert squeeze("".join(c.text for c in strict)) == line


def test_japanese_text_splits_on_its_sentence_terminators() -> None:
    text = japanese(7150)
    chunks = chunk_blocks([Block(text=text)], title="T", target_tokens=350, overlap_tokens=0)
    assert len(chunks) >= 15, "350 tokens is about 350 CJK characters"
    assert all(c.text.endswith(FULL_STOP) for c in chunks), "cut after a full stop, not inside a sentence"
    assert_within_ceilings(chunks, estimate_tokens, 512, 4000)
    assert squeeze("".join(c.text for c in chunks)) == text


def test_japanese_text_without_full_stops_splits_at_commas() -> None:
    text = japanese(7150, stop=False)
    chunks = chunk_blocks([Block(text=text)], title="T", target_tokens=350, overlap_tokens=0)
    assert len(chunks) >= 15
    assert all(c.text.endswith(COMMA) for c in chunks[:-1])
    assert squeeze("".join(c.text for c in chunks)) == text


def test_cjk_sentence_spans() -> None:
    a, b = "a" * 5 + FULL_STOP, "b" * 4 + chr(0xFF01)  # ideographic full stop, full-width exclamation mark
    text = f"{a}{b}English after. Next sentence."
    assert [text[s:e] for s, e in _sentence_spans(text)] == [a, b, "English after.", "Next sentence."]
    mixed = "It costs 5 EUR. " + "c" * 3 + FULL_STOP
    assert [mixed[s:e] for s, e in _sentence_spans(mixed)] == [mixed]  # "c" is not a sentence start


def test_a_base64_blob_respects_both_ceilings() -> None:
    blob = base64.b64encode(random.Random(5).randbytes(150_000)).decode()
    chunks = chunk_blocks(
        [Block(text=f"Attachment follows: {blob}")],
        title="T",
        target_tokens=350,
        overlap_tokens=60,
        count_tokens=strict_counter,
        max_tokens=510,
        max_chars=4000,
    )
    assert_within_ceilings(chunks, strict_counter, 510, 4000)
    assert all(c.token_estimate <= 350 for c in chunks)
    assert blob in squeeze("".join(c.text for c in chunks))


def test_a_long_url_list_is_cut_between_urls() -> None:
    urls = [f"https://example.invalid/docs/section-{i}/page-{i * 7}.html?ref=list&n={i}" for i in range(3000)]
    chunks = chunk_blocks(
        [Block(text=" ".join(urls))], title="Links", target_tokens=350, overlap_tokens=60, count_tokens=strict_counter
    )
    assert_within_ceilings(chunks, strict_counter, 512, 4000)
    pieces = [url for c in chunks for url in c.text.split()]
    assert set(pieces) == set(urls), "every URL survives whole"


def test_embed_text_never_passes_the_model_window() -> None:
    """The title and section prefix count too: a passage that fits the target but not the window with
    its prefix is cut further, never embedded truncated."""
    blocks = [
        Block(text=long_paragraph(30), heading_path=["A very long section heading " * 4]),
        Block(text="\n".join(f"Row {i} | value {i}" for i in range(200)), kind="table", heading_path=["Table"]),
        Block(text="y" * 9000, heading_path=["Blob"]),
    ]
    chunks = chunk_blocks(
        blocks, title="Title " * 20, target_tokens=350, overlap_tokens=60, count_tokens=len, max_tokens=300, max_chars=4000
    )
    assert_within_ceilings(chunks, len, 300, 4000)


def test_char_offsets_point_into_the_document_text() -> None:
    blocks = [
        Block(text=long_paragraph(20), heading_path=["A"]),
        Block(text="Heading B", heading_path=["B"], kind="heading"),
        Block(text=long_paragraph(5, start=100), heading_path=["B"]),
    ]
    full = doc_text(blocks)
    for chunk in chunk_blocks(blocks, title="T", target_tokens=80, overlap_tokens=20):
        assert full[chunk.char_start : chunk.char_end] == chunk.text


def test_giant_sentence_is_cut_into_word_windows() -> None:
    blocks = [Block(text="word " * 500)]
    chunks = chunk_blocks(blocks, title="T", target_tokens=100, overlap_tokens=10)
    assert len(chunks) >= 5
    assert sum(len(c.text.split()) for c in chunks) >= 500
    assert all(c.token_estimate <= 100 for c in chunks)


def test_chunking_is_deterministic(sample_corpus_dir: Path) -> None:
    for path in sorted(sample_corpus_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        parsed = parse(path)
        first: list[ChunkDraft] = chunk_blocks(
            parsed.blocks, title=parsed.title, target_tokens=350, overlap_tokens=60
        )
        second = chunk_blocks(parse(path).blocks, title=parsed.title, target_tokens=350, overlap_tokens=60)
        assert first == second


def test_abbreviations_do_not_end_sentences() -> None:
    text = "Part-time staff accrue leave pro rata, e.g. a three-day week earns 15 days. " * 12
    chunks = chunk_blocks([Block(text=text.strip())], title="T", target_tokens=60, overlap_tokens=0)
    for chunk in chunks:
        assert not chunk.text.startswith("a three-day")
