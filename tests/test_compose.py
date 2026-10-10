"""Answer composition: budget, sentinel, statuses, messages, citations and the extractive digest."""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import pytest

from ragkb.answer import prompts
from ragkb.answer.compose import (
    MESSAGES,
    build_prompt,
    compose,
    compose_provider_error,
    compose_unavailable,
    is_no_answer,
)
from ragkb.answer.prompts import NO_ANSWER, SYSTEM_PROMPT, passage_header
from ragkb.config import Settings
from ragkb.models import AnswerStatus, Chunk, Hit
from ragkb.providers.base import ProviderError, ProviderResult
from ragkb.providers.extractive import ExtractiveProvider

SETTINGS = Settings(MAX_CONTEXT_CHARS=12000, ANSWER_TIMEOUT_S=180)


def make_hit(n: int, text: str, *, doc: str | None = None, title: str = "Handbook", page: int | None = None,
             section: str = "", score: float = 0.03) -> Hit:
    doc_id = doc or f"doc{n}"
    return Hit(
        chunk=Chunk(
            id=f"{doc_id}#{n}", doc_id=doc_id, collection_id="col", ordinal=n, text=text, embed_text=text,
            title=title, section=section, page=page, created_at=datetime.now(UTC),
        ),
        score=score,
    )


class FakeProvider:
    name = "claude_code"
    display_name = "Claude Code"

    def __init__(self, text: str = "", *, error: Exception | None = None, model: str | None = "claude-sonnet-5-5"):
        self.text = text
        self.error = error
        self.model = model
        self.calls: list[dict] = []

    def status(self):  # pragma: no cover - not used by compose
        raise NotImplementedError

    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult:
        self.calls.append({"system": system, "prompt": prompt, "timeout_s": timeout_s, "model": model})
        if self.error:
            raise self.error
        return ProviderResult(text=self.text, model=self.model, elapsed_ms=5)


def run(question: str, hits: list[Hit], provider, *, has_documents: bool = True, settings: Settings = SETTINGS,
        model: str | None = None):
    return compose(question, hits, provider, settings, collection_id="col", knowledge_version="v20261010.3",
                   has_documents=has_documents, model=model)


LEAVE_HITS = [
    make_hit(1, "Full-time employees receive 25 days of paid annual leave per calendar year.", doc="leave",
             title="Leave Policy", page=2, section="Leave > Annual leave", score=0.0325),
    make_hit(2, "Unused annual leave of up to five days can be carried over to the next year.", doc="leave",
             title="Leave Policy", page=3, score=0.031),
    make_hit(3, "New employees may take annual leave after their first three months.", doc="onboarding",
             title="Onboarding Guide", section="Probation", score=0.016),
]


# ---------------------------------------------------------------------------------------------------
# Prompt and budget
# ---------------------------------------------------------------------------------------------------

def test_system_prompt_carries_the_contract():
    for phrase in ("careful knowledge-base assistant", "only the information in the numbered passages",
                   "language of the question", "direct answer in one or two sentences", "headings only",
                   "[2]", "sources list", "do not cover", "Based on the context", f"exactly {NO_ANSWER}",
                   "Never infer that something is absent", "the documents do not mention it, and stop"):
        assert phrase in SYSTEM_PROMPT


def test_prompt_texts_have_no_long_dashes():
    texts = [SYSTEM_PROMPT, prompts.PROMPT_TEMPLATE, prompts.AGENT_PROMPT_TEMPLATE,
             *prompts.DIGEST_LEAD.values(), *prompts.DIGEST_THIN.values()]
    for text in texts:
        assert chr(0x2014) not in text and chr(0x2013) not in text


def test_passage_header_format():
    assert passage_header(1, LEAVE_HITS[0]) == "[1] (Leave Policy, page 2, Leave > Annual leave)"
    assert passage_header(2, LEAVE_HITS[1]) == "[2] (Leave Policy, page 3)"
    assert passage_header(3, LEAVE_HITS[2]) == "[3] (Onboarding Guide, Probation)"


def test_build_prompt_numbers_kept_passages_and_ends_with_the_question():
    prompt, kept = build_prompt("How much leave?", LEAVE_HITS, 12000)
    assert kept == LEAVE_HITS
    assert prompt.index("[1] (Leave Policy") < prompt.index("[2] (") < prompt.index("[3] (") < prompt.index(
        "Question: How much leave?")
    assert NO_ANSWER in prompt


def test_budget_keeps_whole_passages_in_rank_order_and_the_first_always_fits():
    big = [make_hit(i, "x" * 500) for i in range(1, 6)]
    prompt, kept = build_prompt("q", big, 1100)
    assert [h.chunk.id for h in kept] == [big[0].chunk.id, big[1].chunk.id]
    assert "[3]" not in prompt
    prompt, kept = build_prompt("q", big, 10)
    assert len(kept) == 1 and "x" * 500 in prompt


def test_budget_never_cites_a_dropped_passage():
    big = [make_hit(i, f"Fact {i}. " + "y" * 500) for i in range(1, 6)]
    provider = FakeProvider("The facts are known [1]. More is in [2] and [4].")
    answer = run("What are the facts?", big, provider, settings=Settings(MAX_CONTEXT_CHARS=1100))
    assert answer.status == AnswerStatus.answered
    kept_ids = {big[0].chunk.id, big[1].chunk.id}
    assert {c.chunk_id for c in answer.citations} <= kept_ids
    assert "[4]" in answer.answer, "an unknown marker is left as written, never mapped to a dropped passage"
    assert answer.passages_considered == 5
    assert "[3] (" not in provider.calls[0]["prompt"]


# ---------------------------------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------------------------------

def test_answered_fills_every_field():
    provider = FakeProvider("Employees get 25 days a year [2]. New staff wait three months [3].")
    answer = run("How many days of annual leave do employees get?", LEAVE_HITS, provider, model="sonnet")
    assert answer.status == AnswerStatus.answered
    assert answer.answer == "Employees get 25 days a year [1]. New staff wait three months [2]."
    assert answer.message is None
    assert [c.chunk_id for c in answer.citations] == [LEAVE_HITS[1].chunk.id, LEAVE_HITS[2].chunk.id]
    assert answer.passages_considered == 3
    assert answer.language == "en"
    assert answer.knowledge_version == "v20261010.3" and answer.collection_id == "col"
    assert re.fullmatch(r"[0-9a-f]{32}", answer.id)
    assert answer.created_at.tzinfo is not None
    assert answer.provider.name == "claude_code" and answer.provider.model == "claude-sonnet-5-5"
    assert answer.provider.elapsed_ms >= 0
    call = provider.calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["timeout_s"] == 180 and call["model"] == "sonnet"


@pytest.mark.parametrize(
    "reply",
    ["NO_ANSWER", "**NO_ANSWER**", "no_answer", "  \"NO_ANSWER\"  ", "> NO_ANSWER", "# NO_ANSWER", "`NO_ANSWER`",
     "_No_Answer_", "'NO_ANSWER'.", "NO_ANSWER\n\nThe passages are about something else."],
)
def test_decline_with_decoration_is_no_answer(reply: str):
    answer = run("What is the parking policy?", LEAVE_HITS, FakeProvider(reply))
    assert answer.status == AnswerStatus.no_answer
    assert answer.answer is None and answer.citations == []
    assert answer.message == "The knowledge base does not cover this question."


def test_short_uncited_reply_ending_with_the_sentinel_is_a_decline():
    assert is_no_answer("The passages do not mention parking. NO_ANSWER")
    assert not is_no_answer("Leave is 25 days [1]. Unrelated: the token NO_ANSWER is a sentinel [1].")
    assert not is_no_answer("No answer is needed in advance; leave is automatic [1].")
    assert not is_no_answer("")


def test_decline_without_documents_is_no_documents():
    answer = run("Anything?", LEAVE_HITS, FakeProvider("NO_ANSWER"), has_documents=False)
    assert answer.status == AnswerStatus.no_documents
    assert answer.message == "This collection has no documents yet. Upload some to get started."


def test_empty_hits_never_call_the_provider():
    provider = FakeProvider("should not be used")
    empty_collection = run("Anything?", [], provider, has_documents=False)
    assert empty_collection.status == AnswerStatus.no_documents
    nothing_found = run("Anything?", [], provider, has_documents=True)
    assert nothing_found.status == AnswerStatus.no_answer
    assert nothing_found.message == MESSAGES["no_answer"]
    assert provider.calls == []
    assert nothing_found.passages_considered == 0 and nothing_found.provider.name == "claude_code"


def test_empty_model_reply_is_not_an_answer():
    answer = run("How much leave?", LEAVE_HITS, FakeProvider("   "))
    assert answer.status == AnswerStatus.no_answer and answer.answer is None


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (ProviderError("not_signed_in", "x", fix_command="claude auth login"), AnswerStatus.provider_unavailable,
         "Claude Code is not signed in. Run: claude auth login"),
        (ProviderError("not_installed", "x", fix_command="npm install -g x"), AnswerStatus.provider_unavailable,
         "Claude Code is not installed. Install it with: npm install -g x"),
        (ProviderError("timeout", "x"), AnswerStatus.provider_unavailable,
         "Claude Code did not answer within 180 seconds. Try again or ask a narrower question."),
        (ProviderError("failed", "x"), AnswerStatus.provider_unavailable,
         "Claude Code could not produce an answer right now. Try again in a moment."),
        (RuntimeError("adapter bug"), AnswerStatus.provider_unavailable,
         "Claude Code could not produce an answer right now. Try again in a moment."),
    ],
)
def test_provider_errors_become_statuses(error: Exception, status: AnswerStatus, message: str):
    answer = run("How much leave?", LEAVE_HITS, FakeProvider(error=error))
    assert answer.status == status
    assert answer.message == message
    assert answer.answer is None and answer.citations == []
    assert answer.passages_considered == 3 and answer.provider.name == "claude_code"


def test_rate_limited_keeps_the_vendor_detail():
    error = ProviderError("rate_limited", "You've hit your usage limit, resets 9pm (UTC)")
    answer = run("How much leave?", LEAVE_HITS, FakeProvider(error=error))
    assert answer.status == AnswerStatus.rate_limited
    assert answer.message.startswith(
        "Claude Code has reached its usage limit for now. Try again later or choose another provider."
    )
    assert "resets 9pm" in answer.message


def test_compose_unavailable():
    answer = compose_unavailable("How much leave?", collection_id="col", knowledge_version="v1")
    assert answer.status == AnswerStatus.retrieval_unavailable
    assert answer.message == "Search is unavailable right now."
    assert answer.answer is None and answer.citations == [] and answer.passages_considered == 0
    assert answer.provider is None and answer.language == "en"


def test_compose_unavailable_pinned_signature_takes_status_and_message():
    answer = compose_unavailable("How much leave?", collection_id="col", knowledge_version="v1",
                                 status=AnswerStatus.provider_unavailable,
                                 message="No answer provider is available in this installation.")
    assert answer.status == AnswerStatus.provider_unavailable
    assert answer.message == "No answer provider is available in this installation."
    assert answer.answer is None and answer.knowledge_version == "v1" and answer.collection_id == "col"
    default = compose_unavailable("Leave?", collection_id="col", knowledge_version="v1",
                                  status=AnswerStatus.no_documents, message=None)
    assert default.message == MESSAGES["no_documents"]
    with pytest.raises(ValueError):
        compose_unavailable("Leave?", collection_id="col", knowledge_version="v1", status=AnswerStatus.answered,
                            message="An answer.")


def test_compose_provider_error_for_a_provider_that_could_not_be_selected():
    error = ProviderError("not_signed_in", "Codex is not signed in.", fix_command="codex login")
    answer = compose_provider_error("How much leave?", error, provider_name="codex", collection_id="col",
                                    knowledge_version="v1")
    assert answer.status == AnswerStatus.provider_unavailable
    assert answer.message == "Codex is not signed in. Run: codex login"


def test_language_follows_the_question():
    answer = run("¿Cuántos días de vacaciones tienen los empleados?", LEAVE_HITS, FakeProvider("Tienen 25 días [1]."))
    assert answer.language == "es"


def test_passage_text_and_answer_text_never_reach_the_logs(caplog: pytest.LogCaptureFixture):
    secret = "ZEBRA-PASSAGE-7731"
    hits = [make_hit(1, f"The access code is {secret}."), make_hit(2, "Other text.")]
    with caplog.at_level(logging.DEBUG, logger="ragkb"):
        run("What is the access code?", hits, FakeProvider(f"The code is {secret} [1]. Also [9]."))
        run("What is the access code?", hits, FakeProvider(error=ProviderError("failed", "boom")))
        run("What is the access code?", hits, ExtractiveProvider())
    assert secret not in caplog.text
    assert "compose: status=answered" in caplog.text


# ---------------------------------------------------------------------------------------------------
# Extractive provider
# ---------------------------------------------------------------------------------------------------

DIGEST_HITS = [
    make_hit(1, "Annual leave\nFull-time employees receive 25 days of paid annual leave per calendar year.\n"
                "Leave requests go through the HR portal at least two weeks ahead.", doc="leave",
             title="Leave Policy", page=2, score=0.0325),
    make_hit(2, "Unused annual leave of up to five days can be carried over. Remaining days lapse on 31 March.",
             doc="leave", title="Leave Policy", page=3, score=0.031),
    make_hit(3, "New employees may take annual leave after the first three months of employment.",
             doc="onboarding", title="Onboarding Guide", section="People > Probation", score=0.016),
    make_hit(4, "The cafeteria opens at 8 am and closes at 3 pm on weekdays.", doc="facilities",
             title="Facilities", score=0.015),
]


def test_extractive_answers_with_a_readable_cited_digest():
    provider = ExtractiveProvider()
    answer = run("How many days of annual leave do employees get?", DIGEST_HITS, provider)
    assert answer.status == AnswerStatus.answered
    lines = answer.answer.splitlines()
    assert lines[0] == 'Here is what the documents say about "how many days of annual leave do employees get":'
    bullets = [line for line in lines if line.startswith("- ")]
    assert 1 <= len(bullets) <= 5
    assert bullets[0].startswith("- **Leave Policy** (page 2): ")
    assert "25 days" in bullets[0] and "[1]" in bullets[0]
    assert sum(1 for b in bullets if "**Leave Policy**" in b) == 1, "one bullet per document"
    assert not any("cafeteria" in b for b in bullets)
    assert [c.marker for c in answer.citations] == list(range(1, len(answer.citations) + 1))
    assert answer.provider.name == "extractive" and answer.provider.model is None


def test_extractive_digest_markers_are_positions_in_hits():
    digest = ExtractiveProvider().digest("When can new employees take annual leave?", DIGEST_HITS)
    assert "[3]" in digest and "three months" in digest


def test_extractive_thin_coverage_gets_a_closing_line():
    digest = ExtractiveProvider().digest("Can annual leave be paid out in cash at retirement?", DIGEST_HITS)
    assert digest.endswith(prompts.DIGEST_THIN["en"])


def test_extractive_no_answer_when_the_top_score_is_too_low():
    weak = [make_hit(1, DIGEST_HITS[0].chunk.text, score=0.005)]
    answer = run("How many days of annual leave do employees get?", weak, ExtractiveProvider())
    assert answer.status == AnswerStatus.no_answer and answer.answer is None


def test_extractive_no_answer_when_the_question_topic_is_absent():
    answer = run("What is the policy on cryptocurrency investments?", DIGEST_HITS, ExtractiveProvider())
    assert answer.status == AnswerStatus.no_answer
    answer = run("What is the policy on cryptocurrency investments?", DIGEST_HITS, ExtractiveProvider(),
                 has_documents=False)
    assert answer.status == AnswerStatus.no_documents


GENERIC_HITS = [
    make_hit(1, "Company holidays for the year are listed in the office calendar.", doc="holidays",
             title="Office holidays", score=0.033),
    make_hit(2, "The Port Sallow office opens at 08:00 on weekdays.", doc="onboarding",
             title="Onboarding Guide", score=0.032),
    make_hit(3, "The source code of the firmware is kept in the company repository.", doc="it",
             title="IT Security Policy", score=0.031),
    make_hit(4, "Last year the company opened a second office.", doc="history", title="About us", score=0.03),
    make_hit(5, "Staff badges are printed by Facilities.", doc="facilities", title="Facilities", score=0.029),
]


def test_extractive_declines_when_only_generic_words_match():
    provider = ExtractiveProvider()
    # "revenue" is in no passage, and a four-word question tolerates no unmatched topic word.
    assert run("What was the company's revenue last year?", GENERIC_HITS, provider).status == AnswerStatus.no_answer
    # "dress" is tolerated in a six-word question, but no single passage is about the question.
    answer = run("Is there a dress code for staff at the Port Sallow office?", GENERIC_HITS, provider)
    assert answer.status == AnswerStatus.no_answer and answer.answer is None


def test_extractive_tolerates_one_phrasing_word_in_a_long_question():
    hits = [
        make_hit(1, "After probation the notice period is 1 month for employees with less than 3 years of service "
                    "and 3 months for employees with 3 or more years of service.", doc="handbook",
                 title="Employee Handbook", section="Probation and notice", score=0.033),
        make_hit(2, "Expense claims are paid with the next salary.", doc="expenses", title="Expenses", score=0.02),
    ]
    question = "What notice period applies after probation to someone with four years of service?"
    answer = run(question, hits, ExtractiveProvider())
    assert answer.status == AnswerStatus.answered and "3 months" in answer.answer


def test_extractive_question_framing_words_are_not_topic_words():
    hits = [make_hit(1, "CCTV footage is retained for 30 days by Facilities.", doc="retention",
                     title="Data Retention Policy", score=0.033)]
    answer = run("How long do we keep CCTV footage?", hits, ExtractiveProvider())
    assert answer.status == AnswerStatus.answered and "30 days" in answer.answer


def test_extractive_digest_prefers_sentences_with_rare_question_words():
    hits = [
        make_hit(1, "Health insurance\n\nPrivate health insurance covers the employee and their children. Cover "
                    "for a partner is available at a subsidised rate. Dental treatment is included up to EUR 600 "
                    "per person per year.", doc="benefits", title="Benefits", score=0.033),
        make_hit(2, "Damage from water entering through an open service cover is not covered.", doc="faq",
                 title="Product FAQ", score=0.03),
        make_hit(3, "This policy covers business travel and travel insurance.", doc="travel", title="Travel Policy",
                 score=0.028),
    ]
    answer = run("Does the health insurance cover dental treatment?", hits, ExtractiveProvider())
    assert answer.status == AnswerStatus.answered
    # The heading "Health insurance" matches two words too, but "dental" and "treatment" are rarer.
    first_bullet = answer.answer.splitlines()[2]
    assert "EUR 600" in first_bullet and not first_bullet.startswith("- **Benefits**: Health insurance.")


def test_extractive_never_calls_generate():
    provider = ExtractiveProvider()
    with pytest.raises(ProviderError):
        provider.generate("s", "p", timeout_s=1)
    assert run("How many days of annual leave?", DIGEST_HITS, provider).status == AnswerStatus.answered


def test_extractive_lead_line_follows_the_question_language():
    hits = [make_hit(1, "Los empleados a tiempo completo tienen 25 días de vacaciones anuales pagadas.",
                     title="Política de vacaciones")]
    answer = run("¿Cuántos días de vacaciones anuales tienen los empleados?", hits, ExtractiveProvider())
    assert answer.status == AnswerStatus.answered
    assert answer.answer.startswith('Esto es lo que dicen los documentos sobre "')
    assert "[1]" in answer.answer


def test_extractive_handles_lists_and_tables():
    hits = [make_hit(1, "## Allowances\n| Level | Days |\n|---|---|\n| Senior staff | 30 days of annual leave |\n"
                        "- Part-time staff receive annual leave pro rata.", title="Grades")]
    digest = ExtractiveProvider().digest("How much annual leave do senior staff get?", hits)
    assert "- **Grades**: Senior staff, 30 days of annual leave. Part-time staff receive annual leave pro rata [1]." in digest
    assert "|" not in digest and "##" not in digest
