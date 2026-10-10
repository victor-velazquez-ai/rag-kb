"""Embedders: fastembed (local ONNX on CPU) for real use, FakeEmbedder for offline tests.

Both return L2-normalized vectors, so cosine similarity and dot product agree.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Any

from fastembed import TextEmbedding

log = logging.getLogger(__name__)


class EmbedderError(RuntimeError):
    """The embedding model could not be loaded or failed; the message is safe to show."""


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
    with a plain EmbedderError that names supported models."""

    def __init__(self, model_name: str, *, batch_size: int = 32, cache_dir: str | None = None):
        self.name = model_name
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._query_prefix, self._passage_prefix = model_prefixes(model_name)
        self._model: Any = None
        self._lock = threading.Lock()
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
                    try:
                        self._model = TextEmbedding(model_name=self.name, cache_dir=self._cache_dir)
                    except Exception as exc:
                        raise EmbedderError(
                            f"The embedding model {self.name} could not be loaded: {exc}"
                        ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        inputs = [self._passage_prefix + text for text in texts]
        try:
            return [_normalize(v) for v in model.embed(inputs, batch_size=self.batch_size)]
        except Exception as exc:
            raise EmbedderError(f"The embedding model {self.name} failed: {exc}") from exc

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
    """

    def __init__(self, dims: int = 384):
        self.name = f"fake-hash-{dims}"
        self.dims = dims

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
