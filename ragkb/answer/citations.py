"""Citation markers, dense renumbering, snippets and a cheap language guess.

A composed answer cites passages by their position in the prompt, so the raw numbers can have gaps
([1] [4]) or repeat a chunk under two numbers. `densify` turns them into the wire contract: distinct
chunks numbered 1..K in order of first appearance, every marker in the text rewritten in one pass.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable

from ragkb.models import Citation, Hit

logger = logging.getLogger(__name__)

# One bracket group holds one or more numbers: [3] or [1, 4]. Groups written back to back, like [1][3],
# form a run that is rewritten as one unit so repeats inside it collapse.
_GROUP = r"\[\s*\d{1,4}(?:\s*,\s*\d{1,4})*\s*\]"
_RUN_RE = re.compile(rf"(?:{_GROUP})+")
_NUM_RE = re.compile(r"\d+")

SNIPPET_CHARS = 240


def extract_markers(text: str) -> list[int]:
    """Distinct citation numbers in order of first appearance. `[1, 3]` counts as 1 then 3."""
    seen: list[int] = []
    for run in _RUN_RE.finditer(text or ""):
        for num in _NUM_RE.findall(run.group(0)):
            value = int(num)
            if value not in seen:
                seen.append(value)
    return seen


def densify(text: str, kept: list[Hit]) -> tuple[str, list[Citation]]:
    """Renumber the markers of *text* against *kept* (marker n means kept[n - 1]).

    Distinct chunks get 1..K in order of first appearance; two markers that point at the same chunk
    share one number; the text is rewritten in a single pass, so a new number is never matched again
    (rewriting [1] can never corrupt [10]). Markers outside 1..len(kept) are left exactly as written.
    """
    new_for_chunk: dict[str, int] = {}
    citations: list[Citation] = []
    unknown = 0

    def number_for(old: int) -> int | None:
        if not 1 <= old <= len(kept):
            return None
        hit = kept[old - 1]
        chunk_id = hit.chunk.id
        if chunk_id not in new_for_chunk:
            new_for_chunk[chunk_id] = len(citations) + 1
            citations.append(_citation(new_for_chunk[chunk_id], hit))
        return new_for_chunk[chunk_id]

    def swap(match: re.Match[str]) -> str:
        nonlocal unknown
        tokens: list[str] = []
        known_here = 0
        for num in _NUM_RE.findall(match.group(0)):
            old = int(num)
            new = number_for(old)
            if new is None:
                unknown += 1
                token = f"[{old}]"
            else:
                known_here += 1
                token = f"[{new}]"
            if token not in tokens:
                tokens.append(token)
        if known_here == 0:
            return match.group(0)
        return "".join(tokens)

    rewritten = _RUN_RE.sub(swap, text or "")
    if unknown:
        logger.info("densify: %d citation marker(s) outside the %d kept passages left unchanged", unknown, len(kept))
    return rewritten, citations


def _citation(marker: int, hit: Hit) -> Citation:
    chunk = hit.chunk
    return Citation(
        marker=marker,
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        page=chunk.page,
        section=chunk.section,
        snippet=snippet(chunk.text),
        score=hit.score,
    )


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The start of *text* on one line, at most *limit* characters, cut at a word when possible."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    room = max(limit - 3, 1)
    cut = flat[:room]
    space = cut.rfind(" ")
    if space >= room * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "..."


# ---------------------------------------------------------------------------------------------------
# Tokens and language
# ---------------------------------------------------------------------------------------------------

def fold(text: str) -> str:
    """Lowercase and strip accents, so "Días" and "dias" compare equal."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def words(text: str) -> list[str]:
    """Folded word tokens (letters and digits), in order."""
    return _WORD_RE.findall(fold(text))


def stem(token: str) -> str:
    """A deliberately light plural folding: policies -> policy, days -> day, dias -> dia."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Folded (accent-free) function words. They drive the language guess and are removed from questions
# before the extractive provider matches sentences.
STOPWORDS: dict[str, frozenset[str]] = {
    "en": _wordset(
        """
        a about above after against again all also am an and any are as at be because been before being
        below between both but by can could did do does doing down during each either else every few for
        from further get gets got had has have having he her here him his how i if in into is it its itself
        just many may me might more most much must my myself neither no nor not of off on only onto or
        other our out over own per same shall she should so some such than that the their them themselves
        then there these they this those through to too toward towards under until up upon us very via was
        we were what when where whether which while who whom whose why will with within without would yet
        you your yourself
        """
    ),
    "es": _wordset(
        """
        al algo como con cual cuales cuando cuanta cuantas cuanto cuantos de debe deben del donde el ella
        ellos en entre era es esa ese eso esta estan estas este esto estos hay la las le les lo los mas me
        mi mis muy nos nuestra nuestro para pero por porque puede pueden puedo que quien quienes se sea ser
        si sin sobre son su sus tambien tengo tiene tienen tu tus un una unas uno unos usted y ya
        """
    ),
    "pt": _wordset(
        """
        a ao aos as com como da das de deve devem do dos e ela ele em entre era esta estao este isso isto
        ja mais mas meu minha muito na nao nas no nos nossa nosso o onde os ou para pela pelo pode podem
        posso por porque qual quais quando quanta quantas quanto quantos que quem sao se sem ser seu seus
        sobre sua suas tambem tem tenho um uma umas uns voce voces
        """
    ),
    "fr": _wordset(
        """
        a ai au aussi aux avec avez avons ce ces cette combien comment dans de des doit du elle elles en
        est et etre il ils je la le les leur leurs ma mais me mes mon ne nous on ont ou oui par pas peut
        peux plus pour pourquoi quand que quel quelle quelles quels qui quoi sa sans se ses son sont sur
        ta tes ton tres tu un une vos votre vous
        """
    ),
    "de": _wordset(
        """
        aber als am an auch auf bei bin das dass dein dem den der des die du ein eine einem einen einer er
        es fur gibt habe haben hat ich ihr im in ist kann kein keine konnen mein meine mit muss mussen nach
        nicht noch ob oder ohne sich sie sind uber und unser unsere unter viele vom von war warum was welche
        welcher welches wer wie wieviel wir wird wo zu zum zur
        """
    ),
    "it": _wordset(
        """
        agli ai al alla anche che chi ci come con da dal dei del della delle deve devono di dove e gia gli
        ha hanno ho i il in la le lo ma mia mio molto nel nella non nostra nostro o per perche piu possono
        posso puo quale quali quando quanta quante quanti quanto quella quello questa questo si sono su sua
        suo tra tua tuo un una uno
        """
    ),
}

# Words that carry no topic in a question, in any language: requests, the shape of the answer asked
# for ("how long", "how quickly", "the maximum"), vague nouns and light verbs. "How long do we keep CCTV
# footage?" is about CCTV footage; "long" and "keep" only frame it. Matched before and after stemming.
_QUESTION_FILLER = _wordset(
    """
    please tell know need want say says said explain describe give show find list
    long often quickly soon far maximum minimum exactly actually really usually normally generally
    currently still ever already allowed able possible
    someone somebody anyone anybody everyone everybody something anything everything thing way kind sort type
    apply bring keep take make use go come put happen
    favor puedes podrias diga dime explica
    """
)

ALL_STOPWORDS: frozenset[str] = frozenset().union(*STOPWORDS.values()) | _QUESTION_FILLER

# Characters that point at one language, with the weight they add.
_CHAR_HINTS: dict[str, dict[str, float]] = {
    "¿": {"es": 2}, "¡": {"es": 2}, "ñ": {"es": 2},
    "ã": {"pt": 2}, "õ": {"pt": 2}, "ç": {"pt": 1, "fr": 1},
    "ß": {"de": 2}, "ä": {"de": 1}, "ö": {"de": 1}, "ü": {"de": 1},
    "è": {"fr": 1, "it": 1}, "ê": {"fr": 1, "pt": 1}, "ì": {"it": 1}, "ò": {"it": 1},
}


def guess_language(text: str) -> str | None:
    """A cheap guess (en, es, pt, fr, de or it) from function words and telltale characters.

    Returns None when the evidence is thin or tied, which is better than a confident wrong guess.
    """
    tokens = set(words(text))
    if not tokens:
        return None
    scores: dict[str, float] = {lang: float(len(tokens & stop)) for lang, stop in STOPWORDS.items()}
    lowered = (text or "").lower()
    for char, weights in _CHAR_HINTS.items():
        if char in lowered:
            for lang, weight in weights.items():
                scores[lang] += weight
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best_lang, best), (_, second) = ranked[0], ranked[1]
    if best >= 2 and best > second:
        return best_lang
    if best >= 1 and second == 0 and len(tokens) <= 3:
        return best_lang
    return None


def content_tokens(text: str, language: str | None = None) -> list[str]:
    """Distinct stemmed topic words of *text*: function words of *language* (or of every supported
    language when unknown) and question filler removed in any inflection ("applies", "things"), single
    letters dropped, digits kept."""
    stop = (STOPWORDS.get(language, frozenset()) | _QUESTION_FILLER) if language else ALL_STOPWORDS
    out: list[str] = []
    for token in words(text):
        if token in stop or (len(token) < 2 and not token.isdigit()):
            continue
        stemmed = stem(token)
        if stemmed in stop:
            continue
        if stemmed not in out:
            out.append(stemmed)
    return out


def token_set(text: str) -> set[str]:
    """Every stemmed word of *text*, function words included (for matching against content tokens)."""
    return {stem(token) for token in words(text)}


def coverage(question_tokens: Iterable[str], text_tokens: set[str]) -> float:
    """Share of the question's topic words that appear in *text_tokens* (1.0 when there are none)."""
    q = list(question_tokens)
    if not q:
        return 1.0
    return sum(1 for token in q if token in text_tokens) / len(q)
