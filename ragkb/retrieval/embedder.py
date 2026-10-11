"""Embedders: fastembed (local ONNX on CPU) for real use, FakeEmbedder for offline tests.

Both return L2-normalized vectors, so cosine similarity and dot product agree. Both also count tokens
the way their model sees them (``count_tokens``) and say how many fit in one passage (``max_tokens``),
so the chunker can keep every passage inside the model's window: a longer passage would be embedded
truncated, and its tail could never be found.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import queue
import re
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastembed import TextEmbedding

log = logging.getLogger(__name__)

# Batch embedding runs on several single-threaded model sessions at once: on CPU this beats one session
# with many threads, whose threads mostly wait for each other (measured on a 16-core laptop: about 1.7x).
MAX_SESSIONS = 8
# Texts per model run inside a session; small runs keep the activations in the core's cache.
RUN_BATCH = 2
# Sessions beyond the first are released after this many seconds without a batch (each holds the model).
IDLE_RELEASE_S = 60.0


def default_sessions() -> int:
    """Parallel model sessions for batch embedding: half the logical processors, at most MAX_SESSIONS."""
    return max(1, min(MAX_SESSIONS, (os.cpu_count() or 2) // 2))


class EmbedderError(RuntimeError):
    """The embedding model could not be loaded or failed; the message is safe to show."""


# Chinese, Japanese and Korean scripts, CJK punctuation and full-width forms: written without spaces
# between words, so the heuristic count gives each character one token.
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

# The window of most small sentence-embedding models; used when a model's tokenizer cannot be read.
DEFAULT_MAX_TOKENS = 512


def heuristic_token_count(text: str) -> int:
    """Whitespace-separated words plus one per CJK character."""
    cjk = len(_CJK.findall(text))
    words = len((_CJK.sub(" ", text) if cjk else text).split())
    return words + cjk


# (model name pattern, query prefix, passage prefix). Models trained with instructions retrieve better
# when queries and passages carry them; fastembed does not add them itself.
_PREFIXES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"bge-[a-z]+-en-v1\.5", re.IGNORECASE),
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    (re.compile(r"(^|/)(multilingual-)?e5-", re.IGNORECASE), "query: ", "passage: "),
    (re.compile(r"nomic-embed-text", re.IGNORECASE), "search_query: ", "search_document: "),
]


def model_prefixes(model_name: str) -> tuple[str, str]:
    """The (query, passage) prefixes a model expects; empty strings when it needs none."""
    for pattern, query_prefix, passage_prefix in _PREFIXES:
        if pattern.search(model_name):
            return query_prefix, passage_prefix
    return "", ""


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm > 0 else values


# Named in the error for an unknown model: the default, the multilingual option, then larger ones.
SUGGESTED_MODELS = (
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/multilingual-e5-large",
)


def supported_models() -> list[str]:
    """The text embedding models this fastembed version can load (including custom registered ones)."""
    return [str(entry["model"]) for entry in TextEmbedding.list_supported_models()]


class FastEmbedEmbedder:
    """Local embeddings through fastembed. The model loads on first use (a few seconds, and a one-time
    download into the fastembed cache when it is not there yet). An unknown model name fails at once
    with a plain EmbedderError that names supported models.

    Throughput: with ``sessions`` above one (the default is half the logical processors, at most
    MAX_SESSIONS), ``embed`` cuts a batch into model runs of RUN_BATCH texts and hands each run to
    whichever single-threaded ONNX session is free, so several run at once. The extra sessions are made on
    the first large batch and released after IDLE_RELEASE_S idle seconds; each holds a copy of the model
    (about 160 MB for bge-small). Queries always use the first session. ``sessions=1`` gives the plain
    fastembed behaviour: one session, ``threads`` ONNX threads (None: one per physical core) and runs of
    ``batch_size`` texts.
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 64,
        cache_dir: str | None = None,
        threads: int | None = None,
        sessions: int | None = None,
    ):
        self.name = model_name
        self.batch_size = max(1, batch_size)
        self.sessions = max(1, sessions if sessions is not None else default_sessions())
        self.threads = threads if threads is not None else (1 if self.sessions > 1 else None)
        self._cache_dir = cache_dir
        self._query_prefix, self._passage_prefix = model_prefixes(model_name)
        self._model: Any = None
        self._counter: Any = None
        self._max_tokens: int | None = None
        self._lock = threading.Lock()
        # the session pool for batch embedding: idle sessions wait in the queue
        self._pool_lock = threading.Lock()
        self._idle: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._pooled = 0
        self._executor: ThreadPoolExecutor | None = None
        self._busy = 0
        self._last_used = 0.0
        self._reaper: threading.Timer | None = None
        known = {name.casefold(): name for name in supported_models()}
        canonical = known.get(model_name.strip().casefold())
        if canonical is None:
            suggestions = ", ".join(name for name in SUGGESTED_MODELS if name.casefold() in known)
            raise EmbedderError(
                f"The embedding model {model_name} is not supported by this version of fastembed. "
                f"Set RAGKB_EMBED_MODEL to a supported model, for example {suggestions}."
            )
        self.dims = int(TextEmbedding.get_embedding_size(canonical))

    def load(self) -> None:
        """Load the model now instead of on the first request."""
        self._get_model()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._new_session()
        return self._model

    def _new_session(self) -> Any:
        try:
            return TextEmbedding(model_name=self.name, cache_dir=self._cache_dir, threads=self.threads)
        except Exception as exc:
            raise EmbedderError(f"The embedding model {self.name} could not be loaded: {exc}") from exc

    # ------------------------------------------------------------------------------------ tokens

    def _get_counter(self) -> Any:
        """An untruncated, unpadded copy of the model's own tokenizer, or None when it cannot be read
        (counts then fall back to the heuristic). Also settles ``max_tokens``."""
        if self._max_tokens is None:
            model = self._get_model()
            with self._lock:
                if self._max_tokens is None:
                    self._counter, self._max_tokens = self._build_counter(model)
        return self._counter

    def _build_counter(self, model: Any) -> tuple[Any, int]:
        tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
        if tokenizer is None:
            log.warning("the tokenizer of %s is not reachable; token counts are estimated", self.name)
            return None, DEFAULT_MAX_TOKENS
        try:
            counter = type(tokenizer).from_str(tokenizer.to_str())
            counter.no_truncation()
            counter.no_padding()
            window = int((tokenizer.truncation or {}).get("max_length") or DEFAULT_MAX_TOKENS)
            special = len(counter.encode("").ids)  # [CLS] and [SEP], or the model's equivalents
            prefix = len(counter.encode(self._passage_prefix, add_special_tokens=False).ids)
        except Exception as exc:  # noqa: BLE001 - counting must never stop ingestion
            log.warning("the tokenizer of %s could not be copied (%s); token counts are estimated", self.name, exc)
            return None, DEFAULT_MAX_TOKENS
        return counter, max(16, window - special - prefix)

    @property
    def max_tokens(self) -> int:
        """The most tokens one passage can carry without truncation: the model's window minus its
        special tokens and passage prefix (510 for bge-small-en-v1.5, whose window is 512)."""
        self._get_counter()
        assert self._max_tokens is not None
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """Tokens the model's tokenizer makes of ``text``, without special tokens or the passage prefix."""
        counter = self._get_counter()
        if counter is None:
            return heuristic_token_count(text)
        return len(counter.encode(text, add_special_tokens=False).ids)

    # --------------------------------------------------------------------------------- embedding

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = [self._passage_prefix + text for text in texts]
        if self.sessions <= 1 or len(inputs) <= RUN_BATCH:
            return self._run(self._get_model(), inputs)
        # Small runs handed to whichever session is free: on CPUs with fast and slow cores a fixed split
        # would leave the fast ones waiting for the slow ones at the end of every batch.
        runs = [inputs[i : i + RUN_BATCH] for i in range(0, len(inputs), RUN_BATCH)]
        executor = self._open_pool(len(runs))
        try:
            results = list(executor.map(self._run_pooled, runs))
        finally:
            self._close_pool_call()
        return [vector for part in results for vector in part]

    def _run(self, model: Any, inputs: list[str]) -> list[list[float]]:
        run_batch = RUN_BATCH if self.sessions > 1 else self.batch_size
        try:
            return [_normalize(v) for v in model.embed(inputs, batch_size=run_batch)]
        except Exception as exc:
            raise EmbedderError(f"The embedding model {self.name} failed: {exc}") from exc

    def _run_pooled(self, inputs: list[str]) -> list[list[float]]:
        model = self._idle.get()
        try:
            return self._run(model, inputs)
        finally:
            self._idle.put(model)

    def _open_pool(self, wanted: int) -> ThreadPoolExecutor:
        """Make sure ``wanted`` sessions (at most ``sessions``) wait in the pool, and count a call in."""
        primary = self._get_model()
        with self._pool_lock:
            self._busy += 1
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.sessions, thread_name_prefix="embed")
            if self._pooled == 0:
                self._idle.put(primary)
                self._pooled = 1
            missing = min(wanted, self.sessions) - self._pooled
            if missing > 0:
                try:
                    for session in self._executor.map(lambda _: self._new_session(), range(missing)):
                        self._idle.put(session)
                        self._pooled += 1
                except EmbedderError:
                    log.warning("could not open more embedding sessions; continuing with %d", self._pooled)
            return self._executor

    def _close_pool_call(self) -> None:
        with self._pool_lock:
            self._busy -= 1
            self._last_used = time.monotonic()
            if self._pooled > 1 and self._reaper is None:
                self._reaper = threading.Timer(IDLE_RELEASE_S, self._release_idle)
                self._reaper.daemon = True
                self._reaper.start()

    def _release_idle(self) -> None:
        """Drop the sessions beyond the first once no batch has run for IDLE_RELEASE_S seconds."""
        with self._pool_lock:
            self._reaper = None
            idle_for = time.monotonic() - self._last_used
            if self._busy == 0 and idle_for >= IDLE_RELEASE_S:
                while self._pooled > 1:
                    session = self._idle.get()
                    if session is self._model:
                        self._idle.put(session)
                        continue
                    self._pooled -= 1
                log.debug("released the extra embedding sessions")
            elif self._pooled > 1:
                self._reaper = threading.Timer(max(1.0, IDLE_RELEASE_S - idle_for), self._release_idle)
                self._reaper.daemon = True
                self._reaper.start()

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        try:
            vectors = list(model.embed([self._query_prefix + text], batch_size=1))
        except Exception as exc:
            raise EmbedderError(f"The embedding model {self.name} failed: {exc}") from exc
        return _normalize(vectors[0])


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "not",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _fake_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class FakeEmbedder:
    """Deterministic bag-of-hashed-words vectors for tests: texts sharing words get similar vectors.

    Words are lowercased, stop words dropped, plurals folded, and each remaining word adds
    1 + log(count) to one of ``dims`` buckets chosen by a stable hash. No model, no randomness.
    Tokens are counted with the heuristic (whitespace words plus one per CJK character).
    """

    def __init__(self, dims: int = 384, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.name = f"fake-hash-{dims}"
        self.dims = dims
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return heuristic_token_count(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token, count in Counter(_fake_tokens(text)).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "little") % self.dims] += 1.0 + math.log(count)
        if not any(vector):
            vector[0] = 1.0
        return _normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
