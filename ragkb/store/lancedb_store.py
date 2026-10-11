"""LanceDB store for chunks: vectors, cosine search and BM25 full-text search in one embedded table.

One table, ``chunks``, holds every collection. Every read and every destructive statement carries a
collection or document filter, and searches filter before ranking (prefilter), so a search in one
collection can never return or be crowded out by another collection's chunks.

The table is created on the first upsert. An upsert writes all its chunks in one commit: a plain
append when none of their documents has rows yet, otherwise a merge by chunk id. The pipeline upserts a
whole document at once, so a document costs one write. The full-text index on ``text`` is built on the
first upsert. Afterwards it is refreshed (new rows folded in, files compacted) right after any upsert of
REFRESH_MIN_ROWS rows or more, so at most once per document and never while a document is being
written; small upserts are folded in together once REFRESH_PENDING_ROWS rows wait. A refresh costs time
that grows with the table (about 0.3 to 1 second at 20,000 rows) while an append costs about 0.07
seconds, so a folder of small files is not slowed down by one refresh per file. Rows that wait are
still found, because LanceDB also scans unindexed rows.

Writes (upsert with its refresh, deletes) are serialized by one short lock held only around the
LanceDB calls; reads take no lock, run concurrently and always see the latest commit.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from ragkb.models import Chunk, Hit

log = logging.getLogger(__name__)

TABLE_NAME = "chunks"
# An upsert of at least this many rows refreshes the full-text index right away.
REFRESH_MIN_ROWS = 64
# Smaller upserts are folded into the index together once this many rows wait.
REFRESH_PENDING_ROWS = 512
_CHUNK_COLUMNS = [
    "id",
    "doc_id",
    "collection_id",
    "ordinal",
    "text",
    "embed_text",
    "title",
    "section",
    "page",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "token_estimate",
    "knowledge_version",
    "created_at",
]
# Columns added after the first release, with the SQL that fills them in an older table.
_ADDED_COLUMNS = {"page_start": "page", "page_end": "page"}


class StoreError(RuntimeError):
    """The store cannot serve a request; the message is safe to show."""


def _schema(dims: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("collection_id", pa.string()),
            pa.field("ordinal", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("embed_text", pa.string()),
            pa.field("title", pa.string()),
            pa.field("section", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("token_estimate", pa.int32()),
            pa.field("knowledge_version", pa.string()),
            pa.field("created_at", pa.string()),  # ISO 8601, UTC; avoids a time zone database dependency
            pa.field("vector", pa.list_(pa.float32(), dims)),
        ]
    )


def _quote(value: str) -> str:
    """A SQL string literal for a LanceDB filter."""
    return "'" + str(value).replace("'", "''") + "'"


class LanceDBStore:
    """VectorStore on an embedded LanceDB folder."""

    def __init__(self, path: Path, dims: int):
        if dims <= 0:
            raise ValueError("dims must be positive.")
        self.path = Path(path)
        self.dims = dims
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path), read_consistency_interval=timedelta(0))
        self._lock = threading.RLock()
        self._table: Any = None
        self._pending_rows = 0  # rows written since the last full-text refresh

    # ------------------------------------------------------------------------------------- table

    def _table_names(self) -> list[str]:
        result = self._db.list_tables()
        return list(getattr(result, "tables", result))

    def _open(self, create: bool = False) -> Any:
        if self._table is not None:
            return self._table
        with self._lock:
            if self._table is None:
                if TABLE_NAME in self._table_names():
                    table = self._db.open_table(TABLE_NAME)
                    self._add_missing_columns(table)
                    self._table = table
                elif create:
                    self._table = self._db.create_table(TABLE_NAME, schema=_schema(self.dims))
        return self._table

    @staticmethod
    def _add_missing_columns(table: Any) -> None:
        """Give a table written by an older version the columns added since."""
        names = set(table.schema.names)
        missing = {column: sql for column, sql in _ADDED_COLUMNS.items() if column not in names}
        if missing:
            table.add_columns(missing)
            log.info("added columns %s to the chunks table", ", ".join(sorted(missing)))

    def _require_dims(self, table: Any) -> None:
        stored = table.schema.field("vector").type.list_size
        if stored != self.dims:
            raise StoreError(
                f"The index was built with {stored}-dimension vectors, but the embedding model produces "
                f"{self.dims}. Switch back to the previous model, or re-index into a new data folder."
            )

    def _has_text_index(self, table: Any) -> bool:
        return any(
            getattr(index, "index_type", "") == "FTS" and "text" in list(getattr(index, "columns", []))
            for index in table.list_indices()
        )

    def _refresh_text_index(self, table: Any, written: int) -> None:
        """After an upsert of ``written`` rows: build the full-text index on first use; afterwards fold the
        waiting rows in when this upsert was large or enough rows wait (see the module docstring)."""
        self._pending_rows += written
        try:
            if not self._has_text_index(table):
                table.create_index("text", config=FTS(), replace=True)
            elif written >= REFRESH_MIN_ROWS or self._pending_rows >= REFRESH_PENDING_ROWS:
                table.optimize()
            else:
                return
            self._pending_rows = 0
        except Exception as exc:  # noqa: BLE001 - unindexed rows stay searchable; log and carry on
            log.warning("full-text index refresh failed: %s", exc)

    def optimize(self) -> None:
        """Fold every waiting row into the full-text index and compact the table files."""
        table = self._open()
        if table is None:
            return
        with self._lock:
            table.optimize()
            self._pending_rows = 0

    def ensure_text_index(self) -> bool:
        """Create the full-text index if the table exists without one. True when an index exists."""
        table = self._open()
        if table is None:
            return False
        with self._lock:
            if not self._has_text_index(table):
                table.create_index("text", config=FTS(), replace=True)
        return True

    # ------------------------------------------------------------------------------------- writes

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks by id in one commit, then refresh the full-text index when due."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors.")
        if not chunks:
            return
        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dims:
                raise ValueError(f"A vector has {len(vector)} dimensions; this store expects {self.dims}.")
            row = chunk.model_dump(include=set(_CHUNK_COLUMNS))
            created = chunk.created_at if chunk.created_at.tzinfo else chunk.created_at.replace(tzinfo=UTC)
            row["created_at"] = created.isoformat()
            row["vector"] = [float(x) for x in vector]
            rows.append(row)
        data = pa.Table.from_pylist(rows, schema=_schema(self.dims))
        doc_ids = sorted({chunk.doc_id for chunk in chunks})
        with self._lock:
            table = self._open(create=True)
            self._require_dims(table)
            if table.count_rows(f"doc_id IN ({', '.join(_quote(d) for d in doc_ids)})") == 0:
                table.add(data)
            else:
                table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(data)
            self._refresh_text_index(table, len(rows))

    def delete_doc(self, doc_id: str, *, keep_ordinals_below: int | None = None) -> int:
        """Delete a document's chunks; with a bound, only those with ordinal >= the bound (the stale tail)."""
        where = f"doc_id = {_quote(doc_id)}"
        if keep_ordinals_below is not None:
            where += f" AND ordinal >= {int(keep_ordinals_below)}"
        return self._delete(where)

    def delete_collection(self, collection_id: str) -> int:
        return self._delete(f"collection_id = {_quote(collection_id)}")

    def _delete(self, where: str) -> int:
        table = self._open()
        if table is None:
            return 0
        with self._lock:
            result = table.delete(where)
        return int(getattr(result, "num_deleted_rows", 0) or 0)

    # -------------------------------------------------------------------------------------- reads

    def vector_search(
        self, collection_id: str, vector: list[float], k: int, min_score: float = 0.0
    ) -> list[Hit]:
        """Top k chunks of one collection by cosine similarity; drops hits scoring below min_score."""
        table = self._open()
        if table is None or k <= 0:
            return []
        self._require_dims(table)
        if len(vector) != self.dims:
            raise ValueError(
                f"The query vector has {len(vector)} dimensions; this store expects {self.dims}."
            )
        rows = (
            table.search([float(x) for x in vector], vector_column_name="vector", query_type="vector")
            .distance_type("cosine")
            .where(f"collection_id = {_quote(collection_id)}", prefilter=True)
            .select([*_CHUNK_COLUMNS, "_distance"])
            .limit(k)
            .to_list()
        )
        hits: list[Hit] = []
        for row in rows:
            score = 1.0 - float(row["_distance"])
            if score < min_score:
                continue
            hits.append(Hit(chunk=self._chunk(row), score=score, vector_score=score))
        return hits

    def text_search(self, collection_id: str, query: str, k: int) -> list[Hit]:
        """Top k chunks of one collection by BM25 on the chunk text. lexical_rank starts at 1."""
        table = self._open()
        if table is None or k <= 0 or not query.strip():
            return []
        if not self._has_text_index(table):
            self.ensure_text_index()
        rows = (
            table.search(query, query_type="fts", fts_columns="text")
            .where(f"collection_id = {_quote(collection_id)}", prefilter=True)
            .select([*_CHUNK_COLUMNS, "_score"])
            .limit(k)
            .to_list()
        )
        return [
            Hit(chunk=self._chunk(row), score=float(row.get("_score") or 0.0), lexical_rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        """All chunks of one document in ordinal order."""
        return self._chunks_where(f"doc_id = {_quote(doc_id)}")

    def neighbours(self, doc_id: str, ordinal: int, span: int = 1) -> list[Chunk]:
        """The chunks of one document from ``ordinal - span`` to ``ordinal + span``, in ordinal order
        (fewer at the ends of the document)."""
        span = max(0, int(span))
        low, high = int(ordinal) - span, int(ordinal) + span
        return self._chunks_where(f"doc_id = {_quote(doc_id)} AND ordinal >= {low} AND ordinal <= {high}")

    def _chunks_where(self, where: str) -> list[Chunk]:
        table = self._open()
        if table is None:
            return []
        n = table.count_rows(where)
        if n == 0:
            return []
        rows = table.search().where(where).select(_CHUNK_COLUMNS).limit(n).to_list()
        return sorted((self._chunk(r) for r in rows), key=lambda c: c.ordinal)

    def count(self, collection_id: str | None = None) -> int:
        table = self._open()
        if table is None:
            return 0
        if collection_id is None:
            return int(table.count_rows())
        return int(table.count_rows(f"collection_id = {_quote(collection_id)}"))

    def has_documents(self, collection_id: str) -> bool:
        table = self._open()
        if table is None:
            return False
        rows = (
            table.search().where(f"collection_id = {_quote(collection_id)}").select(["id"]).limit(1).to_list()
        )
        return bool(rows)

    @staticmethod
    def _chunk(row: dict[str, Any]) -> Chunk:
        return Chunk(**{key: row.get(key) for key in _CHUNK_COLUMNS})
