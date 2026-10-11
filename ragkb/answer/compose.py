"""Passages + question + provider -> the Answer contract (docs/ARCHITECTURE.md section 5).

The passage budget is applied once and only kept passages are numbered, so a dropped passage can never
be cited. Passages attached as context (``neighbour_of``) get budget only after every ranked passage
that fits, and sit right after their anchor. A decline is an exact sentinel turned into a status, never
text that reads like an answer; with it, ``closest`` lists the best passages so the person can judge.
When a reranker scored the passages and every one is weak by its calibrated band, the decline is made
without spending a model call. Logs carry counts, statuses and timings only, never the question's
passages or the answer text.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import uuid4

from ragkb.answer.citations import closest, densify, extract_markers, guess_language
from ragkb.answer.prompts import (
    NO_ANSWER,
    PASSAGE_SEPARATOR,
    SYSTEM_PROMPT,
    render_passage,
    render_prompt,
)
from ragkb.config import Settings
from ragkb.models import Answer, AnswerStatus, Citation, Hit, ProviderInfo
from ragkb.providers.base import Provider, ProviderError
from ragkb.providers.extractive import ExtractiveProvider
from ragkb.retrieval.base import best_relevance, is_neighbour, ranked_only

logger = logging.getLogger(__name__)

MESSAGES: dict[str, str] = {
    "no_answer": "The knowledge base does not cover this question.",
    "no_documents": "This collection has no documents yet. Upload some to get started.",
    "retrieval_unavailable": "Search is unavailable right now.",
    "not_signed_in": "{provider} is not signed in. Run: {fix_command}",
    "not_signed_in_no_fix": "{provider} is not signed in.",
    "not_installed": "{provider} is not installed. Install it with: {fix_command}",
    "not_installed_no_fix": "{provider} is not installed.",
    "timeout": "{provider} did not answer within {timeout} seconds. Try again or ask a narrower question.",
    "rate_limited": "{provider} has reached its usage limit for now. Try again later or choose another provider.",
    "failed": "{provider} could not produce an answer right now. Try again in a moment.",
}

DISPLAY_NAMES = {"claude_code": "Claude Code", "codex": "Codex", "extractive": "Extractive mode"}

# Leading and trailing decoration a model may wrap the sentinel in: **NO_ANSWER**, "NO_ANSWER", > NO_ANSWER.
_DECORATION = " \t\r\n*_'\"#>`"
# A short reply with no citations that ends with the sentinel is a decline that explained itself first.
_TRAILING_SENTINEL_MAX_CHARS = 300


def is_no_answer(text: str) -> bool:
    """True when *text* is the decline sentinel, tolerating decoration, whitespace and case."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.lstrip(_DECORATION).upper().startswith(NO_ANSWER):
        return True
    tail = stripped.rstrip(_DECORATION + ".").upper()
    return (
        len(stripped) <= _TRAILING_SENTINEL_MAX_CHARS
        and tail.endswith(NO_ANSWER)
        and not extract_markers(stripped)
    )


def build_prompt(question: str, hits: list[Hit], max_context_chars: int) -> tuple[str, list[Hit]]:
    """The user prompt with numbered passages, and the hits it contains, in prompt order.

    Ranked passages are kept whole, in rank order, while their rendered blocks fit *max_context_chars*;
    the first always fits. Passages attached as context then take what is left, each only if it fits
    and only when its anchor was kept, and each is placed right after its anchor."""
    ranked = ranked_only(hits)
    attached: dict[str, list[Hit]] = {}
    for hit in hits:
        if is_neighbour(hit):
            attached.setdefault(str(getattr(hit, "neighbour_of", "")), []).append(hit)

    separator = len(PASSAGE_SEPARATOR)
    kept_ranked: list[Hit] = []
    used = 0
    for hit in ranked:
        cost = len(render_passage(len(kept_ranked) + 1, hit)) + (separator if kept_ranked else 0)
        if kept_ranked and used + cost > max_context_chars:
            break
        kept_ranked.append(hit)
        used += cost

    widest = int("9" * len(str(max(1, len(hits)))))
    extras: dict[str, list[Hit]] = {}
    for anchor in kept_ranked:
        for hit in attached.get(anchor.chunk.id, []):
            context = f"the text just before [{widest}]"
            cost = len(render_passage(widest, hit, context=context)) + separator
            if used + cost > max_context_chars:
                continue
            extras.setdefault(anchor.chunk.id, []).append(hit)
            used += cost

    kept: list[Hit] = []
    for anchor in kept_ranked:
        kept.append(anchor)
        kept.extend(sorted(extras.get(anchor.chunk.id, []), key=lambda h: h.chunk.ordinal))
    return render_prompt(question, kept), kept


def _display(provider: object) -> str:
    name = str(getattr(provider, "name", "provider"))
    return str(getattr(provider, "display_name", None) or DISPLAY_NAMES.get(name, name))


def _now() -> datetime:
    return datetime.now(UTC)


def _answer(
    question: str,
    *,
    status: AnswerStatus,
    collection_id: str,
    knowledge_version: str,
    passages_considered: int,
    provider: ProviderInfo | None,
    answer: str | None = None,
    message: str | None = None,
    citations: list[Citation] | None = None,
    closest_to: list[Hit] | None = None,
) -> Answer:
    """Assemble the contract. *closest_to* are the hits a decline lists as ``closest`` (only when the
    status is no_answer)."""
    return Answer(
        id=uuid4().hex,
        question=question,
        collection_id=collection_id,
        status=status,
        answer=answer if status == AnswerStatus.answered else None,
        message=None if status == AnswerStatus.answered else message,
        citations=citations or [],
        closest=closest(closest_to or []) if status == AnswerStatus.no_answer else [],
        passages_considered=passages_considered,
        provider=provider,
        knowledge_version=knowledge_version,
        language=guess_language(question),
        created_at=_now(),
    )


def _not_covered(has_documents: bool) -> tuple[AnswerStatus, str]:
    if not has_documents:
        return AnswerStatus.no_documents, MESSAGES["no_documents"]
    return AnswerStatus.no_answer, MESSAGES["no_answer"]


def error_message(error: ProviderError, provider_label: str, timeout_s: int | None = None) -> tuple[AnswerStatus, str]:
    """The status and the plain sentence for a provider failure. An error marked ``show_message`` (a
    model the provider does not know, where trying again cannot help) shows its own sentence."""
    if getattr(error, "show_message", False) and error.message:
        return AnswerStatus.provider_unavailable, error.message
    if error.kind == "not_signed_in":
        key = "not_signed_in" if error.fix_command else "not_signed_in_no_fix"
        return AnswerStatus.provider_unavailable, MESSAGES[key].format(
            provider=provider_label, fix_command=error.fix_command
        )
    if error.kind == "not_installed":
        key = "not_installed" if error.fix_command else "not_installed_no_fix"
        return AnswerStatus.provider_unavailable, MESSAGES[key].format(
            provider=provider_label, fix_command=error.fix_command
        )
    if error.kind == "rate_limited":
        message = MESSAGES["rate_limited"].format(provider=provider_label)
        vendor = " ".join((error.message or "").split())
        if vendor and len(vendor) <= 200:
            message += f" The provider said: {vendor.rstrip('.')}."
        return AnswerStatus.rate_limited, message
    if error.kind == "timeout":
        return AnswerStatus.provider_unavailable, MESSAGES["timeout"].format(
            provider=provider_label, timeout=timeout_s if timeout_s is not None else "the allowed"
        )
    return AnswerStatus.provider_unavailable, MESSAGES["failed"].format(provider=provider_label)


def compose(
    question: str,
    hits: list[Hit],
    provider: Provider,
    settings: Settings,
    *,
    collection_id: str,
    knowledge_version: str,
    has_documents: bool,
    model: str | None = None,
) -> Answer:
    """Compose the answer for *question* from *hits* (already ranked) with *provider*.

    Zero hits never reach a model: the status is no_documents or no_answer straight away. The extractive
    provider writes a digest instead of calling a model. When a reranker scored the passages and every
    one is weak, the decline comes before any model call. Provider failures become statuses with a
    plain message; they are never raised. Every no_answer lists the closest passages.
    """
    started = time.monotonic()
    label = _display(provider)
    common = {"collection_id": collection_id, "knowledge_version": knowledge_version, "passages_considered": len(hits)}

    if not hits:
        status, message = _not_covered(has_documents)
        _log(status, provider.name, None, 0, len(hits), 0, 0)
        return _answer(question, status=status, message=message,
                       provider=ProviderInfo(name=provider.name, model=None, elapsed_ms=0), **common)

    if isinstance(provider, ExtractiveProvider):
        answerable = provider.is_answerable(question, hits)
        elapsed = _ms_since(started)
        info = ProviderInfo(name=provider.name, model=None, elapsed_ms=elapsed)
        if not answerable:
            status, message = _not_covered(has_documents)
            _log(status, provider.name, None, elapsed, len(hits), len(hits), 0)
            return _answer(question, status=status, message=message, provider=info, closest_to=hits, **common)
        text, citations = densify(provider.digest(question, hits), hits)
        info.elapsed_ms = _ms_since(started)
        _log(AnswerStatus.answered, provider.name, None, info.elapsed_ms, len(hits), len(hits), len(citations))
        return _answer(question, status=AnswerStatus.answered, answer=text, citations=citations,
                       provider=info, **common)

    if weak_by_reranker(hits):
        status, message = _not_covered(has_documents)
        elapsed = _ms_since(started)
        logger.info("compose: declined before the model call, every reranked passage is weak")
        _log(status, provider.name, None, elapsed, len(hits), 0, 0)
        return _answer(question, status=status, message=message, closest_to=hits,
                       provider=ProviderInfo(name=provider.name, model=None, elapsed_ms=elapsed), **common)

    prompt, kept = build_prompt(question, hits, settings.MAX_CONTEXT_CHARS)
    try:
        result = provider.generate(SYSTEM_PROMPT, prompt, timeout_s=settings.ANSWER_TIMEOUT_S, model=model)
    except ProviderError as exc:
        elapsed = _ms_since(started)
        status, message = error_message(exc, label, settings.ANSWER_TIMEOUT_S)
        logger.warning("compose: provider=%s error kind=%s", provider.name, exc.kind)
        _log(status, provider.name, model, elapsed, len(hits), len(kept), 0)
        return _answer(question, status=status, message=message,
                       provider=ProviderInfo(name=provider.name, model=model, elapsed_ms=elapsed), **common)
    except Exception as exc:  # an adapter bug must still produce an honest status, not a crash
        elapsed = _ms_since(started)
        logger.exception("compose: provider=%s raised %s", provider.name, type(exc).__name__)
        status, message = error_message(ProviderError("failed", type(exc).__name__), label)
        return _answer(question, status=status, message=message,
                       provider=ProviderInfo(name=provider.name, model=model, elapsed_ms=elapsed), **common)

    elapsed = _ms_since(started)
    info = ProviderInfo(name=provider.name, model=result.model or model, elapsed_ms=elapsed)
    text = (result.text or "").strip()
    if not text or is_no_answer(text):
        status, message = _not_covered(has_documents)
        _log(status, provider.name, info.model, elapsed, len(hits), len(kept), 0)
        return _answer(question, status=status, message=message, provider=info, closest_to=hits, **common)

    answer_text, citations = densify(text, kept)
    if not citations:
        logger.warning("compose: provider=%s answered without a usable citation marker", provider.name)
    _log(AnswerStatus.answered, provider.name, info.model, elapsed, len(hits), len(kept), len(citations))
    return _answer(question, status=AnswerStatus.answered, answer=answer_text, citations=citations,
                   provider=info, **common)


_DEFAULT_MESSAGES: dict[AnswerStatus, str] = {
    AnswerStatus.no_answer: MESSAGES["no_answer"],
    AnswerStatus.no_documents: MESSAGES["no_documents"],
    AnswerStatus.retrieval_unavailable: MESSAGES["retrieval_unavailable"],
}


def compose_unavailable(
    question: str,
    *,
    collection_id: str,
    knowledge_version: str,
    status: AnswerStatus = AnswerStatus.retrieval_unavailable,
    message: str | None = None,
) -> Answer:
    """An answer that was never composed: search failed (the default), or no provider could be used.

    *message* is the plain sentence shown to the user; it defaults to the contract's text for
    *status*. An `answered` status is refused, since there is no answer text.
    """
    status = AnswerStatus(status)
    if status == AnswerStatus.answered:
        raise ValueError("compose_unavailable cannot produce an answered status.")
    text = message or _DEFAULT_MESSAGES.get(status) or MESSAGES["failed"].format(provider="The answer provider")
    _log(status, None, None, 0, 0, 0, 0)
    return _answer(
        question,
        status=status,
        message=text,
        collection_id=collection_id,
        knowledge_version=knowledge_version,
        passages_considered=0,
        provider=None,
    )


def compose_provider_error(
    question: str,
    error: ProviderError,
    *,
    provider_name: str,
    collection_id: str,
    knowledge_version: str,
    passages_considered: int = 0,
    model: str | None = None,
) -> Answer:
    """The answer when no provider could be used at all, for example when `ProviderRegistry.select`
    raised because the requested provider is not signed in."""
    status, message = error_message(error, DISPLAY_NAMES.get(provider_name, provider_name))
    _log(status, provider_name, model, 0, passages_considered, 0, 0)
    return _answer(
        question,
        status=status,
        message=message,
        collection_id=collection_id,
        knowledge_version=knowledge_version,
        passages_considered=passages_considered,
        provider=ProviderInfo(name=provider_name, model=model, elapsed_ms=0),
    )


def weak_by_reranker(hits: list[Hit]) -> bool:
    """True when a reranker scored the ranked passages and the calibrated bands call every one weak:
    the knowledge base does not cover the question, so no model call is spent on it."""
    ranked = ranked_only(hits)
    if not any(hit.rerank_score is not None for hit in ranked):
        return False
    return best_relevance(ranked) == "weak"


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _log(
    status: AnswerStatus,
    provider: str | None,
    model: str | None,
    elapsed_ms: int,
    passages: int,
    kept: int,
    citations: int,
) -> None:
    logger.info(
        "compose: status=%s provider=%s model=%s elapsed_ms=%d passages=%d kept=%d citations=%d",
        status.value, provider, model, elapsed_ms, passages, kept, citations,
    )
