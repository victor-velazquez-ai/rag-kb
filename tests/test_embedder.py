"""Embedders: token counting (heuristic and the model's tokenizer), the window, and the session pool that
runs batch embedding on several single-threaded sessions at once.

The pool is tested offline against a stand-in for fastembed's TextEmbedding. Tests marked live load the
real model (run them with RAGKB_LIVE_TESTS=1)."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from ragkb.retrieval import embedder as embedder_module
from ragkb.retrieval.embedder import (
    DEFAULT_MAX_TOKENS,
    RUN_BATCH,
    FakeEmbedder,
    FastEmbedEmbedder,
    default_sessions,
    heuristic_token_count,
)

live = pytest.mark.skipif(os.environ.get("RAGKB_LIVE_TESTS") != "1", reason="set RAGKB_LIVE_TESTS=1 to load the model")
CJK_A, CJK_B, FULL_STOP = chr(0x65E5), chr(0x672C), chr(0x3002)


class StandInModel:
    """Stands in for fastembed's TextEmbedding: vectors from text length, a record of runs, and a check
    that no session is used by two threads at once."""

    sessions: ClassVar[list[StandInModel]] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None):
        self.threads = threads
        self.runs: list[int] = []
        self.batch_sizes: set[int] = set()
        self.in_use = threading.Lock()
        self.model = SimpleNamespace(tokenizer=None)
        StandInModel.sessions.append(self)

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return [{"model": "stand-in/model"}]

    @classmethod
    def get_embedding_size(cls, name: str) -> int:
        return 3

    def embed(self, texts: Any, batch_size: int = 256) -> Iterator[list[float]]:
        texts = list(texts)
        assert self.in_use.acquire(blocking=False), "a session was used by two threads at once"
        try:
            self.runs.append(len(texts))
            self.batch_sizes.add(batch_size)
            time.sleep(0.02)
            vectors = [[float(len(t)), 1.0, 0.0] for t in texts]
        finally:
            self.in_use.release()
        yield from vectors


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> type[StandInModel]:
    StandInModel.sessions = []
    monkeypatch.setattr(embedder_module, "TextEmbedding", StandInModel)
    return StandInModel


def expected(text: str) -> list[float]:
    norm = math.sqrt(len(text) ** 2 + 1)
    return [len(text) / norm, 1 / norm, 0.0]


# ------------------------------------------------------------------------------------------- tokens


def test_heuristic_token_count() -> None:
    assert heuristic_token_count("") == 0
    assert heuristic_token_count("three short words") == 3
    assert heuristic_token_count(CJK_A + CJK_B + FULL_STOP) == 3, "one per CJK character"
    assert heuristic_token_count(f"ab {CJK_A}{CJK_B} cd") == 4


def test_fake_embedder_counts_tokens_and_has_a_window() -> None:
    fake = FakeEmbedder()
    assert fake.max_tokens == DEFAULT_MAX_TOKENS == 512
    assert fake.count_tokens("one two three") == 3
    assert FakeEmbedder(max_tokens=64).max_tokens == 64


def test_an_unreadable_tokenizer_falls_back_to_the_heuristic(stand_in: type[StandInModel]) -> None:
    model = FastEmbedEmbedder("stand-in/model", sessions=1)
    assert model.max_tokens == DEFAULT_MAX_TOKENS
    assert model.count_tokens("four words right here") == 4


def test_default_sessions_follow_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder_module.os, "cpu_count", lambda: 22)
    assert default_sessions() == 8
    monkeypatch.setattr(embedder_module.os, "cpu_count", lambda: 4)
    assert default_sessions() == 2
    monkeypatch.setattr(embedder_module.os, "cpu_count", lambda: None)
    assert default_sessions() == 1


# --------------------------------------------------------------------------------------- the pool


def test_a_batch_is_split_across_single_threaded_sessions_in_order(stand_in: type[StandInModel]) -> None:
    model = FastEmbedEmbedder("stand-in/model", sessions=4)
    texts = [f"passage {'x' * i}" for i in range(40)]
    assert model.embed(texts) == [expected(t) for t in texts], "order is kept"
    assert len(stand_in.sessions) == 4
    assert all(s.threads == 1 and s.batch_sizes == {RUN_BATCH} for s in stand_in.sessions)
    assert sum(sum(s.runs) for s in stand_in.sessions) == 40
    assert all(s.runs for s in stand_in.sessions), "every session took a share"


def test_small_batches_and_queries_use_the_first_session_only(stand_in: type[StandInModel]) -> None:
    model = FastEmbedEmbedder("stand-in/model", sessions=4)
    assert model.embed(["one"]) == [expected("one")]
    model.embed_query("a question")
    assert len(stand_in.sessions) == 1


def test_one_session_keeps_the_plain_behaviour(stand_in: type[StandInModel]) -> None:
    model = FastEmbedEmbedder("stand-in/model", sessions=1, batch_size=16)
    texts = [f"t{i}" for i in range(40)]
    assert model.embed(texts) == [expected(t) for t in texts]
    [session] = stand_in.sessions
    assert session.threads is None and session.batch_sizes == {16}


def test_concurrent_batches_share_the_pool_safely(stand_in: type[StandInModel]) -> None:
    model = FastEmbedEmbedder("stand-in/model", sessions=3)
    results: dict[int, list[list[float]]] = {}
    batches = {n: [f"doc {n} passage {'y' * i}" for i in range(17)] for n in range(4)}

    def work(n: int) -> None:
        results[n] = model.embed(batches[n])

    threads = [threading.Thread(target=work, args=(n,)) for n in batches]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert all(results[n] == [expected(t) for t in batches[n]] for n in batches)
    assert len(stand_in.sessions) == 3


def test_extra_sessions_are_released_when_idle(stand_in: type[StandInModel], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder_module, "IDLE_RELEASE_S", 0.1)
    model = FastEmbedEmbedder("stand-in/model", sessions=4)
    model.embed([f"text {i}" for i in range(16)])
    assert model._pooled == 4
    deadline = time.monotonic() + 5
    while model._pooled > 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert model._pooled == 1
    model.embed([f"again {i}" for i in range(16)])  # the pool grows back on the next large batch
    assert model._pooled == 4 and len(stand_in.sessions) == 7


# --------------------------------------------------------------------------------------------- live


@live
def test_real_tokenizer_counts_without_truncation() -> None:
    model = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    assert model.max_tokens == 510, "a window of 512 minus [CLS] and [SEP]"
    assert model.count_tokens("hello world") == 2
    long_text = "The quick brown fox jumps over the lazy dog. " * 200
    assert model.count_tokens(long_text) == 2000, "ten tokens a sentence, not cut at the window"


@live
def test_pooled_vectors_match_a_single_session() -> None:
    texts = [f"Passage {i} about expense claims, receipts and approvals for trip {i}." for i in range(24)]
    pooled = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", sessions=4).embed(texts)
    single = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", sessions=1).embed(texts)
    for a, b in zip(pooled, single, strict=True):
        assert sum(x * y for x, y in zip(a, b, strict=True)) > 0.9999
