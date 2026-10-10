"""Chunker: pack structural blocks into passages of about ``target_tokens``.

Rules, in order of importance:

1. A block no longer than the target is never split. A longer block is split on its natural seams:
   sentences for paragraphs, items for lists, lines for code, and rows for tables (each part of a
   split table repeats the header row). A single sentence or row longer than the target is cut into
   word windows as a last resort.
2. Consecutive pieces are packed until the next one would pass the target. A heading always travels
   with the piece that follows it, so a passage never ends on a dangling heading.
3. Sections are packed together toward the target, but a new top-level section starts a new chunk
   once the current one holds a quarter of the target, and a new sub-section does once it holds half.
   So small sibling sections share a chunk while unrelated topics rarely do.
4. When a chunk is cut for size inside one section of prose, the next chunk starts with the trailing
   sentences of the previous one, up to ``overlap_tokens`` (at most a third of the target).

``section`` is " > ".join(heading_path) when the chunk lies in one section; when it spans several, it
is their common parent path, or, when they share no parent, their top-level headings joined by ", "
(for example "Devices, Networks"). Intro text before the first heading does not affect the label.
``text`` is clean display text; ``embed_text`` adds "title > section: " in front for the embedding.
``char_start`` and ``char_end`` are offsets into the document text formed by joining the block texts
with blank lines. Tokens are estimated as words times 1.3. Output is fully deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ragkb.models import Block

TOKENS_PER_WORD = 1.3
BLOCK_SEPARATOR = "\n\n"
_PROSE = ("paragraph", "list")


def estimate_tokens(text: str) -> int:
    """Approximate token count: words times 1.3, rounded up."""
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    embed_text: str
    section: str
    page: int | None
    char_start: int
    char_end: int
    token_estimate: int


@dataclass
class _Piece:
    text: str
    start: int
    end: int
    tokens: int
    kind: str
    path: tuple[str, ...]
    page: int | None
    synthetic: bool = False  # text is built (a table part with its header), not a slice of the document


@dataclass
class _Unit:
    pieces: list[_Piece] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(p.tokens for p in self.pieces)

    @property
    def path(self) -> tuple[str, ...]:
        return self.pieces[-1].path

    @property
    def first(self) -> _Piece:
        return self.pieces[0]

    @property
    def last(self) -> _Piece:
        return self.pieces[-1]


def chunk_blocks(
    blocks: list[Block], *, title: str, target_tokens: int, overlap_tokens: int
) -> list[ChunkDraft]:
    """Pack blocks into chunk drafts. See the module docstring for the rules."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    overlap_budget = max(0, min(overlap_tokens, target_tokens // 3))
    new_topic_tokens = max(1, target_tokens // 4)
    new_subsection_tokens = max(1, target_tokens // 2)

    doc_text, pieces = _pieces(blocks, target_tokens)
    units = _units(pieces)

    drafts: list[ChunkDraft] = []
    current: list[_Unit] = []
    overlap_start: int | None = None
    overlap_tokens_used = 0

    for unit in units:
        if current:
            own = sum(u.tokens for u in current)
            previous_path = current[-1].path
            section_change = unit.path != previous_path
            new_topic = unit.path[:1] != previous_path[:1]
            too_big = overlap_tokens_used + own + unit.tokens > target_tokens
            if (
                too_big
                or (new_topic and own >= new_topic_tokens)
                or (section_change and own >= new_subsection_tokens)
            ):
                drafts.append(_draft(doc_text, current, overlap_start, title))
                overlap_start, overlap_tokens_used = None, 0
                if not section_change:
                    overlap_start, overlap_tokens_used = _overlap(
                        doc_text, current, unit, overlap_budget, target_tokens
                    )
                current = []
        current.append(unit)
    if current:
        drafts.append(_draft(doc_text, current, overlap_start, title))
    return drafts


# --------------------------------------------------------------------------------------------------
# blocks -> pieces -> units
# --------------------------------------------------------------------------------------------------


def _pieces(blocks: list[Block], target: int) -> tuple[str, list[_Piece]]:
    texts: list[str] = []
    pieces: list[_Piece] = []
    offset = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if texts:
            offset += len(BLOCK_SEPARATOR)
        base = offset
        texts.append(text)
        offset += len(text)
        path = tuple(block.heading_path)
        tokens = estimate_tokens(text)
        if tokens <= target:
            pieces.append(_Piece(text, base, base + len(text), tokens, block.kind, path, block.page))
            continue
        if block.kind == "table":
            pieces.extend(_split_table(text, base, target, path, block.page))
            continue
        if block.kind == "code":
            spans = _line_spans(text)
        elif block.kind == "list":
            spans = []
            for s, e in _line_spans(text):
                if estimate_tokens(text[s:e]) <= target:
                    spans.append((s, e))
                else:
                    spans.extend((s + a, s + b) for a, b in _sentence_spans(text[s:e]))
        else:
            spans = _sentence_spans(text)
        for start, end in _group(text, spans, target):
            piece_text = text[start:end]
            pieces.append(
                _Piece(
                    piece_text,
                    base + start,
                    base + end,
                    estimate_tokens(piece_text),
                    block.kind,
                    path,
                    block.page,
                )
            )
    return BLOCK_SEPARATOR.join(texts), pieces


def _units(pieces: list[_Piece]) -> list[_Unit]:
    """Fuse each run of heading pieces with the piece that follows it."""
    units: list[_Unit] = []
    pending: list[_Piece] = []
    for piece in pieces:
        pending.append(piece)
        if piece.kind != "heading":
            units.append(_Unit(pending))
            pending = []
    if pending:
        units.append(_Unit(pending))
    return units


def _split_table(text: str, base: int, target: int, path: tuple[str, ...], page: int | None) -> list[_Piece]:
    lines = text.split("\n")
    header, rows = lines[0], lines[1:]
    if not rows:
        return [
            _Piece(text[s:e], base + s, base + e, estimate_tokens(text[s:e]), "paragraph", path, page)
            for s, e in _group(text, [(0, len(text))], target)
        ]
    header_tokens = estimate_tokens(header)
    budget = max(target - header_tokens, target // 4, 1)
    row_spans: list[tuple[int, int]] = []
    position = len(header) + 1
    for row in rows:
        row_spans.append((position, position + len(row)))
        position += len(row) + 1
    pieces: list[_Piece] = []
    for start, end in _group(text, row_spans, budget):
        body = text[start:end]
        piece_text = f"{header}\n{body}"
        pieces.append(
            _Piece(
                piece_text, base + start, base + end, estimate_tokens(piece_text), "table", path, page, True
            )
        )
    return pieces


def _group(text: str, spans: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Group consecutive spans while they fit the target; spans that alone exceed it are cut into
    word windows first. Returns (start, end) offsets into ``text``."""
    fitted: list[tuple[int, int]] = []
    for start, end in spans:
        if estimate_tokens(text[start:end]) <= target:
            fitted.append((start, end))
        else:
            fitted.extend(_word_windows(text, start, end, target))
    groups: list[tuple[int, int]] = []
    group_start: int | None = None
    group_end = 0
    group_tokens = 0
    for start, end in fitted:
        tokens = estimate_tokens(text[start:end])
        if group_start is not None and group_tokens + tokens > target:
            groups.append((group_start, group_end))
            group_start = None
        if group_start is None:
            group_start, group_tokens = start, 0
        group_end = end
        group_tokens += tokens
    if group_start is not None:
        groups.append((group_start, group_end))
    return groups


def _word_windows(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    words = [(start + m.start(), start + m.end()) for m in re.finditer(r"\S+", text[start:end])]
    size = max(1, int(target / TOKENS_PER_WORD))
    return [(words[i][0], words[min(i + size, len(words)) - 1][1]) for i in range(0, len(words), size)]


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for line in text.split("\n"):
        if line.strip():
            lead = len(line) - len(line.lstrip())
            spans.append((position + lead, position + len(line.rstrip())))
        position += len(line) + 1
    return spans


_ABBREVIATIONS = {
    "e.g",
    "i.e",
    "etc",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "no",
    "nos",
    "vs",
    "st",
    "inc",
    "ltd",
    "co",
    "corp",
    "approx",
    "fig",
    "jr",
    "sr",
    "dept",
    "est",
    "p",
    "pp",
    "vol",
    "ref",
    "al",
    "cf",
    "ca",
}
_CANDIDATE = re.compile(r"[.!?][\"')\]\u201d\u2019]*(?=\s)|\n")
_LAST_WORD = re.compile(r"(\S+)$")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) offsets. Splits after . ! ? when the next word starts a sentence, and at
    line breaks; skips common abbreviations and single-letter initials."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _CANDIDATE.finditer(text):
        if m.group() == "\n":
            cut = m.start()
        else:
            cut = m.end()
            if m.group()[0] == ".":
                word = _LAST_WORD.search(text[max(start, m.start() - 40) : m.start()])
                last = word.group(1).lower().lstrip("(\"'") if word else ""
                if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                    continue
            following = text[cut:].lstrip()[:1]
            if following and not (following.isupper() or following.isdigit() or following in "\"'([-*\u201c\u2018\u2022"):
                continue
        _add_span(text, start, cut, spans)
        start = cut
    _add_span(text, start, len(text), spans)
    return spans


def _add_span(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    segment = text[start:end]
    if segment.strip():
        lead = len(segment) - len(segment.lstrip())
        spans.append((start + lead, start + len(segment.rstrip())))


# --------------------------------------------------------------------------------------------------
# overlap and drafts
# --------------------------------------------------------------------------------------------------


def _overlap(
    doc_text: str, previous: list[_Unit], next_unit: _Unit, budget: int, target: int
) -> tuple[int | None, int]:
    """Start offset and token count of the trailing sentences carried into the next chunk."""
    tail, head = previous[-1].last, next_unit.first
    if budget <= 0 or tail.synthetic or head.synthetic or tail.kind not in _PROSE or head.kind not in _PROSE:
        return None, 0
    previous_tokens = sum(u.tokens for u in previous)
    budget = min(budget, previous_tokens // 2, int(1.5 * target) - next_unit.tokens)
    if budget <= 0:
        return None, 0
    start: int | None = None
    used = 0
    pieces = [p for u in previous for p in u.pieces]
    for piece in reversed(pieces):
        if piece.synthetic or piece.kind not in _PROSE:
            break
        spans = _sentence_spans(doc_text[piece.start : piece.end])
        for s, e in reversed(spans):
            tokens = estimate_tokens(doc_text[piece.start + s : piece.start + e])
            if used + tokens > budget:
                return start, used
            start, used = piece.start + s, used + tokens
    return start, used


def _draft(doc_text: str, units: list[_Unit], overlap_start: int | None, title: str) -> ChunkDraft:
    pieces = [p for u in units for p in u.pieces]
    segments: list[str] = []
    run_start: int | None = overlap_start
    run_end = 0
    for piece in pieces:
        if piece.synthetic:
            if run_start is not None:
                segments.append(doc_text[run_start:run_end])
                run_start = None
            segments.append(piece.text)
            continue
        if run_start is None:
            run_start = piece.start
        run_end = piece.end
    if run_start is not None:
        segments.append(doc_text[run_start:run_end])
    text = BLOCK_SEPARATOR.join(s for s in segments if s.strip())

    section = _section_label([u.path for u in units])
    page = next((p.page for p in pieces if p.page is not None), None)
    prefix = " > ".join(part for part in (title.strip(), section) if part)
    embed_text = f"{prefix}: {text}" if prefix else text
    return ChunkDraft(
        text=text,
        embed_text=embed_text,
        section=section,
        page=page,
        char_start=overlap_start if overlap_start is not None else pieces[0].start,
        char_end=pieces[-1].end,
        token_estimate=estimate_tokens(text),
    )


def _section_label(paths: list[tuple[str, ...]]) -> str:
    named = [p for p in paths if p]
    if not named:
        return ""
    prefix = list(named[0])
    for path in named[1:]:
        n = 0
        while n < len(prefix) and n < len(path) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    if prefix:
        return " > ".join(prefix)
    tops = list(dict.fromkeys(p[0] for p in named))
    return ", ".join(tops)
