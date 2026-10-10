"""The no-model provider: a readable digest of the best sentences, so the knowledge base works with no
subscription at all. It is also what the MCP tools hand to the caller's own assistant.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ragkb import __version__
from ragkb.answer.citations import content_tokens, coverage, guess_language, token_set
from ragkb.answer.prompts import DIGEST_BULLET, DIGEST_LEAD, DIGEST_PAGE_WORD, DIGEST_THIN
from ragkb.models import Hit, ProviderStatus
from ragkb.providers.base import ProviderError, ProviderResult

# When to answer. Fused scores alone cannot decline (hybrid search returns its top matches for any
# question), so the passages must also be about the question. Each topic word of the question weighs
# by its rarity across the retrieved passages, an IDF weight 1 + ln((1 + N) / (1 + df)) where df counts
# the passages that contain it: a generic word found in most passages ("company", "year", "office")
# weighs little, a specific one ("CCTV", "probation") a lot, and a word no passage contains weighs the
# most. The rules, all required:
# 1. The best fused score reaches EXTRACTIVE_MIN_SCORE (RRF scores: 1 / (60 + rank + 1) per list).
EXTRACTIVE_MIN_SCORE = 0.01
# 2. The rarest words are covered: a topic word that no passage contains is tolerated only once per
#    EXTRACTIVE_WORDS_PER_MISS topic words. A long question may carry one phrasing word the documents
#    put differently ("four years" against "3 or more years"); in a short one every word is the topic.
EXTRACTIVE_WORDS_PER_MISS = 5
# 3. The passages together hold at least this share of the question's weight...
EXTRACTIVE_MIN_COVERAGE = 0.6
# 4. ...and one passage alone holds at least this share: an answer sits in a passage about the question,
#    not in generic words scattered across unrelated ones.
EXTRACTIVE_MIN_FOCUS = 0.5
# Below this share of topic words covered by the digest's own sentences, a closing line says so.
DIGEST_THIN_COVERAGE = 0.75
MAX_BULLETS = 5
MAX_SENTENCES_PER_BULLET = 2
MIN_SENTENCE_CHARS = 12
MAX_SENTENCE_CHARS = 320

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[¿¡]?[A-Z0-9ÁÉÍÓÚÑÀÈÌÒÙÂÊÎÔÛÄÖÜÇ])")
_LIST_MARK = re.compile(r"^(?:[-*+•]|\d{1,3}[.)])\s+")
_HEADING_MARK = re.compile(r"^#{1,6}\s+")
_TABLE_RULE = re.compile(r"^[\s|:\-+]+$")


@dataclass
class _Sentence:
    text: str
    hit_index: int      # 0-based position in hits; the marker is hit_index + 1
    position: int       # order inside the passage
    tokens: set[str] = field(default_factory=set)
    overlap: int = 0        # how many of the question's topic words the sentence contains
    weight: float = 0.0     # the same words weighted by rarity (term_weights), used for ranking


def _clip(sentence: str) -> str:
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return sentence
    cut = sentence[: MAX_SENTENCE_CHARS - 3]
    space = cut.rfind(" ")
    if space > MAX_SENTENCE_CHARS * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + "..."


def _units(text: str) -> list[str]:
    """Readable units of a passage: sentences of prose (wrapped lines joined back), list items and
    table rows on their own, headings and table rules dropped."""
    out: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            joined = " ".join(prose)
            out.extend(part.strip() for part in _SENTENCE_SPLIT.split(joined))
            prose.clear()

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if _HEADING_MARK.match(line):
            flush()
            continue
        if line.count("|") >= 2:
            flush()
            if not _TABLE_RULE.match(line):
                cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
                out.append(", ".join(cells))
            continue
        if _LIST_MARK.match(line):
            flush()
            out.append(_LIST_MARK.sub("", line))
            continue
        prose.append(line)
    flush()
    return [_finish(_clip(" ".join(unit.split()))) for unit in out if len(unit.strip()) >= MIN_SENTENCE_CHARS]


def _finish(sentence: str) -> str:
    """End every unit with punctuation, so a table row or list item reads as a sentence when joined."""
    return sentence if sentence[-1:] in ".!?:;" else sentence + "."


def _with_marker(sentence: str, marker: int) -> str:
    if sentence[-1:] in ".!?" and not sentence.endswith("..."):
        return f"{sentence[:-1]} [{marker}]{sentence[-1]}"
    return f"{sentence} [{marker}]"


def term_weights(terms: list[str], passages: list[set[str]]) -> dict[str, float]:
    """IDF weight of each term across *passages* (token sets): 1 + ln((1 + N) / (1 + df))."""
    n = len(passages)
    return {term: 1.0 + math.log((1 + n) / (1 + sum(1 for p in passages if term in p))) for term in terms}


def _topic(question: str) -> str:
    topic = " ".join(question.split()).strip().lstrip("¿¡").rstrip("?!.;: ")
    if len(topic) > 120:
        topic = topic[:117].rstrip() + "..."
    if len(topic) > 1 and topic[0].isupper() and topic[1].islower():
        topic = topic[0].lower() + topic[1:]
    return topic.replace('"', "'")


class ExtractiveProvider:
    """Always available. The composer calls `digest`; `generate` is not used."""

    name = "extractive"
    display_name = "Extractive mode"

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            installed=True,
            signed_in=True,
            version=__version__,
            detail="Built in. Shows the most relevant sentences from the documents, with no AI model.",
            fix_command=None,
            models=[],
        )

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult:
        raise ProviderError("failed", "The extractive provider has no model; compose calls digest() instead.")

    # -- assessment -------------------------------------------------------------------------------

    def is_answerable(self, question: str, hits: list[Hit]) -> bool:
        """True when the passages are about the question, by the four rules at the top of this module."""
        if not hits:
            return False
        if max(hit.score for hit in hits) < EXTRACTIVE_MIN_SCORE:
            return False
        terms = content_tokens(question, guess_language(question))
        if not terms:
            return True
        passages = [token_set(f"{hit.chunk.title} {hit.chunk.section} {hit.chunk.text}") for hit in hits]
        missing = [term for term in terms if not any(term in passage for passage in passages)]
        if len(missing) > len(terms) // EXTRACTIVE_WORDS_PER_MISS:
            return False
        weights = term_weights(terms, passages)
        total = sum(weights.values())
        found = set().union(*passages)
        if sum(weight for term, weight in weights.items() if term in found) < EXTRACTIVE_MIN_COVERAGE * total:
            return False
        focus = max(sum(weight for term, weight in weights.items() if term in passage) for passage in passages)
        return focus >= EXTRACTIVE_MIN_FOCUS * total

    # -- digest -----------------------------------------------------------------------------------

    def digest(self, question: str, hits: list[Hit]) -> str:
        """Markdown: a lead line, one bullet per source document (up to MAX_BULLETS) with its best one or
        two sentences and [n] markers (n = 1-based position in *hits*), and a closing line when the
        sentences cover the question thinly."""
        language = guess_language(question) or "en"
        if language not in DIGEST_LEAD:
            language = "en"
        question_tokens = content_tokens(question, guess_language(question))
        qset = set(question_tokens)
        weights = term_weights(
            question_tokens, [token_set(f"{hit.chunk.title} {hit.chunk.section} {hit.chunk.text}") for hit in hits]
        )

        # Group candidate sentences by document, documents in rank order of their first hit.
        by_doc: dict[str, list[_Sentence]] = {}
        doc_hit: dict[str, int] = {}
        for index, hit in enumerate(hits):
            doc_id = hit.chunk.doc_id
            doc_hit.setdefault(doc_id, index)
            bucket = by_doc.setdefault(doc_id, [])
            for position, text in enumerate(_units(hit.chunk.text)):
                tokens = token_set(text)
                matched = qset & tokens
                bucket.append(_Sentence(text, index, position, tokens, len(matched),
                                        sum(weights[t] for t in matched)))

        any_overlap = any(s.overlap for sentences in by_doc.values() for s in sentences)
        bullets: list[str] = []
        covered: set[str] = set()
        for doc_id in sorted(by_doc, key=lambda d: doc_hit[d]):
            if len(bullets) >= MAX_BULLETS:
                break
            sentences = by_doc[doc_id]
            if not sentences:
                continue
            ranked = sorted(sentences, key=lambda s: (-s.weight, s.hit_index, s.position))
            best = ranked[0]
            if any_overlap and best.overlap == 0:
                continue
            chosen = [best]
            for candidate in ranked[1:]:
                if len(chosen) >= MAX_SENTENCES_PER_BULLET:
                    break
                if candidate.overlap >= max(1, best.overlap - 1) and candidate.text != best.text:
                    chosen.append(candidate)
            chosen.sort(key=lambda s: (s.hit_index, s.position))
            for sentence in chosen:
                covered |= qset & sentence.tokens
            bullets.append(self._bullet(hits[doc_hit[doc_id]], chosen, language))
            if not any_overlap and len(bullets) >= 3:
                break

        lines = [DIGEST_LEAD[language].format(topic=_topic(question)), ""]
        lines.extend(bullets)
        if bullets and coverage(question_tokens, covered) < DIGEST_THIN_COVERAGE:
            lines.extend(["", DIGEST_THIN[language]])
        return "\n".join(lines).strip()

    @staticmethod
    def _bullet(first_hit: Hit, chosen: list[_Sentence], language: str) -> str:
        chunk = first_hit.chunk
        title = " ".join((chunk.title or "Untitled").replace("*", "").split())
        if chunk.page is not None:
            where = f" ({DIGEST_PAGE_WORD[language]} {chunk.page})"
        elif chunk.section.strip():
            last = chunk.section.split(">")[-1].strip().replace("*", "")
            where = f" ({last})" if last and last != title else ""
        else:
            where = ""
        parts: list[str] = []
        for i, sentence in enumerate(chosen):
            last_of_hit = i == len(chosen) - 1 or chosen[i + 1].hit_index != sentence.hit_index
            parts.append(_with_marker(sentence.text, sentence.hit_index + 1) if last_of_hit else sentence.text)
        return DIGEST_BULLET.format(title=title, where=where, text=" ".join(parts))
