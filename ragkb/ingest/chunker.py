"""Chunker: pack structural blocks into passages of about ``target_tokens``.

Rules, in order of importance:

1. Hard ceilings. No passage's display text is longer than ``max_chars`` characters, and no passage's
   ``embed_text`` has more than ``max_tokens`` tokens (the embedding model's window), so nothing is ever
   embedded truncated. Tokens are counted with ``count_tokens``: the embedding model's own tokenizer in
   the pipeline, a words estimate by default.
2. A block that fits is never split. A longer block is split on its natural seams: sentences for
   paragraphs (after . ! ? and the CJK and full-width terminators), items for lists, lines for code,
   and rows for tables (each part of a split table repeats the header row). A piece still too long is
   cut at clause marks followed by a space (and the CJK clause marks), then between words, then at any
   comma, semicolon, colon, slash or bar, and as the last resort into fixed windows of characters.
3. Consecutive pieces are packed until the next one would pass the target. A heading always travels
   with the piece that follows it, so a passage never ends on a dangling heading.
4. Pages. A passage never holds text from two pages: each page starts a new passage, so a page shorter
   than the target is a passage of its own and every passage's page is exact. A heading that ends a page
   travels with the text that follows it on the next page; pages are counted from the text, not from
   such a heading.
5. Sections are packed together toward the target, but a new top-level section starts a new chunk
   once the current one holds a quarter of the target, and a new sub-section does once it holds half.
   So small sibling sections share a chunk while unrelated topics rarely do.
6. When a chunk is cut for size inside one section of prose on one page, the next chunk starts with the
   trailing sentences of the previous one, up to ``overlap_tokens`` (at most a third of the target), and
   never past one and a half targets or the ceilings.

``section`` is " > ".join(heading_path) when the chunk lies in one section; when it spans several, it
is their common parent path, or, when they share no parent, their top-level headings joined by ", "
(for example "Devices, Networks"). Intro text before the first heading does not affect the label.
``text`` is clean display text; ``embed_text`` adds "title > section: " in front for the embedding.
``page``, ``page_start`` and ``page_end`` are the passage's page (all equal for paged formats, None
otherwise). ``char_start`` and ``char_end`` are offsets into the document text formed by joining the
block texts with blank lines. Output is fully deterministic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ragkb.models import Block

TOKENS_PER_WORD = 1.3
BLOCK_SEPARATOR = "\n\n"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_CHARS = 4000
# The title and the section label are each cut to this many characters in front of ``embed_text``.
PREFIX_PART_CHARS = 200
_MIN_LIMIT = 8
# A passage whose estimated embed_text is this close to the window is counted exactly.
_EXACT_MARGIN = 8
_PROSE = ("paragraph", "list")

TokenCounter = Callable[[str], int]

_CJK_RANGES = (
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # half-width and full-width forms
)
_CJK = re.compile("[" + "".join(f"{chr(low)}-{chr(high)}" for low, high in _CJK_RANGES) + "]")


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer: words times 1.3, rounded up, plus one per CJK
    character (those scripts put no spaces between words)."""
    cjk = len(_CJK.findall(text))
    words = len((_CJK.sub(" ", text) if cjk else text).split())
    return math.ceil(words * TOKENS_PER_WORD) + cjk


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    embed_text: str
    section: str
    page: int | None
    char_start: int
    char_end: int
    token_estimate: int
    page_start: int | None = None
    page_end: int | None = None


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
    def chars(self) -> int:
        return sum(len(p.text) for p in self.pieces) + len(BLOCK_SEPARATOR) * (len(self.pieces) - 1)

    @property
    def path(self) -> tuple[str, ...]:
        return self.pieces[-1].path

    @property
    def page(self) -> int | None:
        """The page of the unit's text; a heading's page counts only when the unit has nothing else."""
        return _content_page(self.pieces)

    @property
    def first(self) -> _Piece:
        return self.pieces[0]

    @property
    def last(self) -> _Piece:
        return self.pieces[-1]


def _content_page(pieces: list[_Piece]) -> int | None:
    for piece in pieces:
        if piece.kind != "heading" and piece.page is not None:
            return piece.page
    return next((p.page for p in pieces if p.page is not None), None)


def chunk_blocks(
    blocks: list[Block],
    *,
    title: str,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[ChunkDraft]:
    """Pack blocks into chunk drafts. See the module docstring for the rules."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    if max_tokens <= 0 or max_chars <= 0:
        raise ValueError("max_tokens and max_chars must be positive.")
    chunker = _Chunker(
        title=title,
        target=target_tokens,
        overlap=overlap_tokens,
        count=count_tokens or estimate_tokens,
        max_tokens=max_tokens,
        max_chars=max_chars,
    )
    return chunker.run(blocks)


def _embed_prefix(title: str, section: str) -> str:
    parts = (title.strip()[:PREFIX_PART_CHARS], section[:PREFIX_PART_CHARS])
    return " > ".join(part for part in parts if part)


class _Chunker:
    def __init__(
        self, *, title: str, target: int, overlap: int, count: TokenCounter, max_tokens: int, max_chars: int
    ):
        self.title = title
        self.target = target
        self.count = count
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.overlap_budget = max(0, min(overlap, target // 3))
        self.new_topic_tokens = max(1, target // 4)
        self.new_subsection_tokens = max(1, target // 2)
        self._prefix_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------------------------ limits

    def _room(self, section: str) -> int:
        """Tokens left for text in one passage once the "title > section:" prefix is counted."""
        used = self._prefix_tokens.get(section)
        if used is None:
            prefix = _embed_prefix(self.title, section)
            used = self.count(prefix + ":") if prefix else 0
            self._prefix_tokens[section] = used
        return self.max_tokens - used

    def limit(self, path: tuple[str, ...]) -> int:
        """The packing target for text under ``path``: the target, or less when the window is smaller."""
        return max(_MIN_LIMIT, min(self.target, self._room(" > ".join(path))))

    def ceiling(self, path: tuple[str, ...]) -> int:
        """What a passage may reach with its overlap: one and a half targets, within the window."""
        return max(_MIN_LIMIT, min(int(1.5 * self.target), self._room(" > ".join(path))))

    # ------------------------------------------------------------------------------------ packing

    def run(self, blocks: list[Block]) -> list[ChunkDraft]:
        doc_text, pieces = self._pieces(blocks)
        units = _units(pieces)

        drafts: list[ChunkDraft] = []
        current: list[_Unit] = []
        current_page: int | None = None
        overlap_start: int | None = None
        overlap_tokens = 0
        overlap_chars = 0

        for unit in units:
            if current:
                own = sum(u.tokens for u in current)
                own_chars = sum(u.chars for u in current) + len(BLOCK_SEPARATOR) * (len(current) - 1)
                previous_path = current[-1].path
                section_change = unit.path != previous_path
                new_topic = unit.path[:1] != previous_path[:1]
                page_change = current_page is not None and unit.page is not None and unit.page != current_page
                limit = min(self.limit(previous_path), self.limit(unit.path))
                too_big = (
                    overlap_tokens + own + unit.tokens > limit
                    or overlap_chars + own_chars + len(BLOCK_SEPARATOR) + unit.chars > self.max_chars
                )
                if (
                    page_change
                    or too_big
                    or (new_topic and own >= self.new_topic_tokens)
                    or (section_change and own >= self.new_subsection_tokens)
                ):
                    self._emit(drafts, doc_text, current, overlap_start)
                    overlap_start, overlap_tokens, overlap_chars = None, 0, 0
                    if not section_change and not page_change:
                        overlap_start, overlap_tokens, overlap_chars = self._overlap(doc_text, current, unit)
                    current, current_page = [], None
            current.append(unit)
            if current_page is None:
                current_page = unit.page
        if current:
            self._emit(drafts, doc_text, current, overlap_start)
        return drafts

    def _emit(self, drafts: list[ChunkDraft], doc_text: str, units: list[_Unit], overlap_start: int | None) -> None:
        """Append the draft for ``units``; when it breaks a ceiling (token sums are estimates for some
        tokenizers, and a heading adds to its unit), drop the overlap, then halve, then cut finer."""
        draft, embed_tokens = self._draft(doc_text, units, overlap_start)
        if (embed_tokens <= self.max_tokens and len(draft.text) <= self.max_chars) or (
            len(units) == 1 and len(units[0].pieces) == 1 and len(units[0].first.text) <= 1
        ):
            drafts.append(draft)
            return
        if overlap_start is not None:
            self._emit(drafts, doc_text, units, None)
            return
        if len(units) > 1:
            half = len(units) // 2
            self._emit(drafts, doc_text, units[:half], None)
            self._emit(drafts, doc_text, units[half:], None)
            return
        *headings, piece = units[0].pieces
        smaller = self._shrink(piece)
        if len(smaller) <= 1 and headings:
            for part in [*headings, piece]:
                self._emit(drafts, doc_text, [_Unit([part])], None)
            return
        for index, part in enumerate(smaller):
            self._emit(drafts, doc_text, [_Unit([*headings, part] if index == 0 else [part])], None)

    def _shrink(self, piece: _Piece) -> list[_Piece]:
        """Cut one piece into parts of at most half its tokens and characters."""
        text = piece.text
        if len(text) <= 1:
            return [piece]
        spans = self._group(text, [(0, len(text))], max(1, piece.tokens // 2), max(1, len(text) // 2))
        if piece.synthetic:
            return [
                _Piece(text[s:e], piece.start, piece.end, n, piece.kind, piece.path, piece.page, True)
                for s, e, n in spans
            ]
        return [
            _Piece(text[s:e], piece.start + s, piece.start + e, n, piece.kind, piece.path, piece.page)
            for s, e, n in spans
        ]

    # ----------------------------------------------------------------------------------- pieces

    def _pieces(self, blocks: list[Block]) -> tuple[str, list[_Piece]]:
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
            limit = self.limit(path)
            tokens = self.count(text) if len(text) <= self.max_chars else None  # a longer block never fits
            if tokens is not None and tokens <= limit:
                pieces.append(_Piece(text, base, base + len(text), tokens, block.kind, path, block.page))
                continue
            if block.kind == "table":
                pieces.extend(self._split_table(text, base, limit, path, block.page))
                continue
            if block.kind == "code":
                spans = _line_spans(text)
            elif block.kind == "list":
                spans = []
                for s, e in _line_spans(text):
                    if e - s <= self.max_chars and self.count(text[s:e]) <= limit:
                        spans.append((s, e))
                    else:
                        spans.extend((s + a, s + b) for a, b in _sentence_spans(text[s:e]))
            else:
                spans = _sentence_spans(text)
            for start, end, n in self._group(text, spans, limit, self.max_chars):
                pieces.append(
                    _Piece(text[start:end], base + start, base + end, n, block.kind, path, block.page)
                )
        return BLOCK_SEPARATOR.join(texts), pieces

    def _split_table(
        self, text: str, base: int, limit: int, path: tuple[str, ...], page: int | None
    ) -> list[_Piece]:
        lines = text.split("\n")
        header, rows = lines[0], lines[1:]
        header_tokens = self.count(header)
        if not rows or header_tokens > limit // 2 or len(header) > self.max_chars // 2:
            # no header worth repeating: split by lines like any other text
            return [
                _Piece(text[s:e], base + s, base + e, n, "table", path, page)
                for s, e, n in self._group(text, _line_spans(text), limit, self.max_chars)
            ]
        row_spans: list[tuple[int, int]] = []
        position = len(header) + 1
        for row in rows:
            if row.strip():
                row_spans.append((position, position + len(row)))
            position += len(row) + 1
        pieces: list[_Piece] = []
        for start, end, n in self._group(text, row_spans, limit - header_tokens, self.max_chars - len(header) - 1):
            piece_text = f"{header}\n{text[start:end]}"
            pieces.append(
                _Piece(piece_text, base + start, base + end, header_tokens + n, "table", path, page, True)
            )
        return pieces

    def _group(
        self, text: str, spans: list[tuple[int, int]], limit: int, char_limit: int
    ) -> list[tuple[int, int, int]]:
        """Group consecutive spans while they fit ``limit`` tokens and ``char_limit`` characters; a span
        that alone does not fit is cut finer first. Returns (start, end, tokens) in ``text``."""
        fitted: list[tuple[int, int, int]] = []
        for start, end in spans:
            n = self.count(text[start:end]) if end - start <= char_limit else None
            if n is not None and n <= limit:
                fitted.append((start, end, n))
            else:
                fitted.extend(self._hard_split(text, start, end, limit, char_limit, 0))
        groups: list[tuple[int, int, int]] = []
        group_start: int | None = None
        group_end = 0
        group_tokens = 0
        for start, end, n in fitted:
            if group_start is not None and (group_tokens + n > limit or end - group_start > char_limit):
                groups.append((group_start, group_end, group_tokens))
                group_start = None
            if group_start is None:
                group_start, group_tokens = start, 0
            group_end = end
            group_tokens += n
        if group_start is not None:
            groups.append((group_start, group_end, group_tokens))
        return groups

    def _hard_split(
        self, text: str, start: int, end: int, limit: int, char_limit: int, level: int
    ) -> list[tuple[int, int, int]]:
        """Cut a span that has no sentence boundary, trying each seam in ``_CUTS`` in turn, and fixed
        windows last. Returns (start, end, tokens) parts that each fit."""
        if level >= len(_CUTS):
            return self._windows(text, start, end, limit, char_limit)
        parts = _cut(text, start, end, _CUTS[level])
        if len(parts) <= 1:
            return self._hard_split(text, start, end, limit, char_limit, level + 1)
        out: list[tuple[int, int, int]] = []
        for a, b in parts:
            n = self.count(text[a:b]) if b - a <= char_limit else None
            if n is not None and n <= limit:
                out.append((a, b, n))
            else:
                out.extend(self._hard_split(text, a, b, limit, char_limit, level + 1))
        return out

    def _windows(self, text: str, start: int, end: int, limit: int, char_limit: int) -> list[tuple[int, int, int]]:
        """Fixed windows of characters, each within ``char_limit`` characters and ``limit`` tokens."""
        out: list[tuple[int, int, int]] = []
        position = start
        while position < end:
            while position < end and text[position].isspace():
                position += 1
            if position >= end:
                break
            size = min(char_limit, end - position)
            n = self.count(text[position : position + size])
            while n > limit and size > 1:
                size = max(1, min(size - 1, size * limit // n))
                n = self.count(text[position : position + size])
            stop = position + size
            trimmed = stop
            while trimmed > position and text[trimmed - 1].isspace():
                trimmed -= 1
            if trimmed != stop:
                n = self.count(text[position:trimmed])
            out.append((position, trimmed, n))
            position = stop
        return out

    # ----------------------------------------------------------------------------- overlap, drafts

    def _overlap(self, doc_text: str, previous: list[_Unit], next_unit: _Unit) -> tuple[int | None, int, int]:
        """Start offset, tokens and characters of the trailing sentences carried into the next chunk."""
        tail, head = previous[-1].last, next_unit.first
        if (
            self.overlap_budget <= 0
            or tail.synthetic
            or head.synthetic
            or tail.kind not in _PROSE
            or head.kind not in _PROSE
        ):
            return None, 0, 0
        previous_tokens = sum(u.tokens for u in previous)
        budget = min(self.overlap_budget, previous_tokens // 2, self.ceiling(next_unit.path) - next_unit.tokens)
        char_budget = self.max_chars - next_unit.chars - len(BLOCK_SEPARATOR)
        if budget <= 0 or char_budget <= 0:
            return None, 0, 0
        end = tail.end
        start: int | None = None
        used = 0
        pieces = [p for u in previous for p in u.pieces]
        for piece in reversed(pieces):
            if piece.synthetic or piece.kind not in _PROSE:
                break
            for s, e in reversed(_sentence_spans(doc_text[piece.start : piece.end])):
                candidate = piece.start + s
                if end - candidate > char_budget:
                    return start, used, (end - start if start is not None else 0)
                tokens = self.count(doc_text[candidate : piece.start + e])
                if used + tokens > budget:
                    return start, used, (end - start if start is not None else 0)
                start, used = candidate, used + tokens
        return start, used, (end - start if start is not None else 0)

    def _draft(self, doc_text: str, units: list[_Unit], overlap_start: int | None) -> tuple[ChunkDraft, int]:
        """The draft for ``units`` and the token count of its ``embed_text``."""
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
        pages = [p.page for p in pieces if p.page is not None and p.kind != "heading"] or [
            p.page for p in pieces if p.page is not None
        ]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
        prefix = _embed_prefix(self.title, section)
        embed_text = f"{prefix}: {text}" if prefix else text
        # Token counts add up across a space, so the prefix and the text are counted apart (the prefix
        # once per section); near the window the embedded text is counted exactly.
        text_tokens = self.count(text)
        embed_tokens = text_tokens + (self.max_tokens - self._room(section)) if prefix else text_tokens
        if prefix and embed_tokens > self.max_tokens - _EXACT_MARGIN:
            embed_tokens = self.count(embed_text)
        draft = ChunkDraft(
            text=text,
            embed_text=embed_text,
            section=section,
            page=page_start,
            char_start=overlap_start if overlap_start is not None else pieces[0].start,
            char_end=pieces[-1].end,
            token_estimate=text_tokens,
            page_start=page_start,
            page_end=page_end,
        )
        return draft, embed_tokens


# --------------------------------------------------------------------------------------------------
# blocks -> units, spans
# --------------------------------------------------------------------------------------------------


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


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for line in text.split("\n"):
        if line.strip():
            lead = len(line) - len(line.lstrip())
            spans.append((position + lead, position + len(line.rstrip())))
        position += len(line) + 1
    return spans


# Seams for a span with no sentence boundary, tried in order; a span is cut just after each match.
_CJK_CLAUSE_MARKS = "".join(map(chr, (0x3001, 0xFF0C, 0xFF1B, 0xFF1A)))  # ideographic and full-width , ; :
_CUTS = (
    re.compile(f"[,;:](?=\\s)|[{_CJK_CLAUSE_MARKS}]"),  # clauses
    re.compile(r"\s+"),  # words
    re.compile(r"[,;:/|&?=]"),  # inside long tokens: URLs, paths, comma-separated values
)


def _cut(text: str, start: int, end: int, seam: re.Pattern[str]) -> list[tuple[int, int]]:
    """Split text[start:end] just after every match of ``seam``; parts are trimmed, empty ones dropped."""
    parts: list[tuple[int, int]] = []
    position = start
    for match in seam.finditer(text, start, end):
        if match.end() > position:
            _add_span(text, position, match.end(), parts)
            position = match.end()
    _add_span(text, position, end, parts)
    return parts


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
# CJK full stop, full-width ! ? and full stop, half-width ideographic full stop: end a sentence with no
# space after them. The closers are quotes and brackets that may follow a terminator.
_CJK_TERMINATORS = "".join(map(chr, (0x3002, 0xFF01, 0xFF1F, 0xFF0E, 0xFF61)))
_CLOSERS = "\"')\\]" + "".join(map(chr, (0x201D, 0x2019, 0x300D, 0x300F, 0x3011, 0xFF09)))
_CANDIDATE = re.compile(f"[.!?][{_CLOSERS}]*(?=\\s)|[{_CJK_TERMINATORS}][{_CLOSERS}]*|\\n")
_SENTENCE_OPENERS = "\"'([-*" + "".join(map(chr, (0x201C, 0x2018, 0x2022)))
_NEXT_VISIBLE = re.compile(r"\s*(\S)")
_LAST_WORD = re.compile(r"(\S+)$")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) offsets. Splits after . ! ? when the next word starts a sentence, after
    the CJK and full-width terminators, and at line breaks; skips common abbreviations and initials."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _CANDIDATE.finditer(text):
        mark = m.group()
        if mark == "\n":
            cut = m.start()
        else:
            cut = m.end()
            if mark[0] in ".!?":
                if mark[0] == ".":
                    word = _LAST_WORD.search(text, max(start, m.start() - 40), m.start())
                    last = word.group(1).lower().lstrip("(\"'") if word else ""
                    if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                        continue
                visible = _NEXT_VISIBLE.match(text, cut)
                following = visible.group(1) if visible else ""
                if following and not (
                    following.isupper()
                    or following.isdigit()
                    or following in _SENTENCE_OPENERS
                    or _CJK.match(following)
                ):
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
