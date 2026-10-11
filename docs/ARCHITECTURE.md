# Architecture and contracts

This document is the contract between the parts. Each part is built against these interfaces and data shapes; change them here first.

## 1. Package layout

```
ragkb/
  __init__.py          version
  config.py            Settings (env prefix RAGKB_), data folders
  models.py            Pydantic models shared by every layer
  cli.py               `ragkb` command: serve, ingest, ask, doctor, eval
  ingest/
    parsers.py         file -> Parsed document (blocks with heading path and page)
    chunker.py         blocks -> chunks (about 350 tokens, within the embedding model's window, overlap, metadata kept)
    pipeline.py        upload -> parse -> chunk -> embed -> store, status and version
  store/
    base.py            VectorStore protocol, MetadataStore protocol
    lancedb_store.py   chunks and vectors, full-text index, hybrid search
    sqlite_meta.py     collections, documents, jobs, feedback, audit, version counter
  retrieval/
    base.py            Embedder, Reranker protocols; RankedHit; RRF; exact terms; near-duplicate drop; relevance bands
    embedder.py        fastembed embedder (parallel sessions, the model's own token count); FakeEmbedder for tests
    reranker.py        fastembed cross-encoder reranker; NoReranker; reranker_for(settings)
    service.py         Retriever: query -> ranked hits, within the composer's budget when asked
  answer/
    prompts.py         the composition prompt and the extractive digest format
    compose.py         passages + question + provider -> Answer (contract below)
    citations.py       dense renumbering, dedupe by source, snippet building
  providers/
    base.py            Provider protocol, ProviderResult, ProviderStatus
    claude_code.py     Claude Code CLI adapter
    codex.py           Codex CLI adapter
    extractive.py      no-model provider
    registry.py        detection, selection, status for the UI and doctor
  server/
    app.py             FastAPI app factory, static UI mount, lifespan
    routes/            health.py, collections.py, documents.py, search.py, ask.py, feedback.py, providers.py
    deps.py            shared singletons (settings, stores, retriever, providers)
  mcp/
    server.py          MCP tools over streamable HTTP at /mcp
  evals/
    golden.py          golden set format and loader
    scorers.py         recall@k, MRR, citation rate, false-answer rate, decline rate
    run.py             offline and live runs, report
  ui/                  index.html, app.js, styles.css (no build step)
samples/
  corpus/              a small fictional corpus for demos and tests
  golden.yaml          the sample golden set
tests/
docs/
```

## 2. Settings (`ragkb/config.py`, env prefix `RAGKB_`)

| Setting | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | Root for `lancedb/`, `uploads/`, `meta.sqlite`, `logs/` |
| `HOST`, `PORT` | `127.0.0.1`, `8765` | Server bind |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name (downloaded on first use, about 130 MB) |
| `RERANKER` | `local` | `local` (the cross-encoder below, downloaded on first use, about 90 MB) or `none` |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed reranker |
| `TOP_K` | `6` | the least number of passages given to the composer |
| `CANDIDATES` | `30` | fused candidates considered before reranking and the budget |
| `NEIGHBOUR_EXPANSION` | `true` | attach the previous and next passage of the strongest hits when they fit the budget |
| `MIN_SCORE` | `0.0` | vector-score floor, 0 to 1; lexical hits bypass it |
| `MAX_CONTEXT_CHARS` | `12000` | composition budget |
| `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS` | `350`, `60` | chunking targets, counted with the embedding model's own tokenizer |
| `MAX_CHUNK_CHARS` | `4000` | hard ceiling on a passage's display text |
| `SKIP_HIDDEN_SHEETS` | `true` | leave hidden spreadsheet sheets out of the index |
| `INGEST_WORKERS` | `2` | parallel ingestion jobs |
| `EMBED_BATCH_SIZE` | `64` | passages per embedding batch (progress and cancel checks happen between batches) |
| `MAX_UPLOAD_MB` | `50` | per file |
| `PROVIDER` | `auto` | `auto`, `claude_code`, `codex`, `extractive` |
| `CLAUDE_BIN`, `CODEX_BIN` | auto-detected | explicit paths to the CLIs |
| `CLAUDE_MODEL` | `sonnet` | passed to `--model` |
| `CODEX_MODEL` | empty (the CLI's own default) | passed to `-m` when set |
| `CODEX_REASONING` | `medium` | passed as `-c model_reasoning_effort=...` |
| `ANSWER_TIMEOUT_S` | `180` | provider call timeout |
| `MCP_ENABLED` | `true` | mount `/mcp` |
| `MCP_PUBLIC` | `false` | `/mcp` answers any `Host`, not only loopback |
| `API_PUBLIC` | `false` | `/api` answers any `Host`, not only loopback (`GET /api/health` always answers) |
| `UI_ENABLED` | `true` | mount the UI at `/` |
| `LOG_LEVEL` | `INFO` | |

A test (`tests/test_config_documented.py`) asserts that every setting, and every `RAGKB_` variable the code reads from the environment (the test-only `RAGKB_FAKE_EMBEDDER`), appears in `.env.example` with its default, and the reverse.

## 3. Data model (`ragkb/models.py`)

```python
class Collection(BaseModel):
    id: str; name: str; description: str = ""; created_at: datetime

class DocumentStatus(str, Enum): queued, processing, ready, failed, cancelled

class Document(BaseModel):
    id: str                      # uuid4 hex
    collection_id: str
    title: str                   # a label for people; titles may repeat
    filename: str
    source_path: str = ""        # the identity in the collection: the relative path from a folder or zip upload, else the filename
    media_type: str              # application/pdf, text/markdown, ...
    size_bytes: int
    content_hash: str            # sha256 of the raw bytes
    pages: int | None
    chunk_count: int = 0
    status: DocumentStatus
    error: str | None = None
    uploaded_at: datetime
    indexed_at: datetime | None = None
    knowledge_version: str | None = None
    metadata: dict = {}

class Block(BaseModel):          # parser output
    text: str; heading_path: list[str] = []; page: int | None = None; kind: str = "paragraph"  # paragraph, heading, table, list, code

class Chunk(BaseModel):          # store row
    id: str                      # f"{doc_id}#{ordinal}"
    doc_id: str; collection_id: str; ordinal: int
    text: str                    # display text
    embed_text: str              # "title > section: text" used for the vector
    title: str; section: str     # section = " > ".join(heading_path)
    page: int | None             # the first page of the chunk
    page_start: int | None; page_end: int | None   # every page the chunk touches
    char_start: int; char_end: int; token_estimate: int
    knowledge_version: str; created_at: datetime

class Hit(BaseModel):
    chunk: Chunk
    score: float                 # the fused score that orders the results
    vector_score: float | None   # raw cosine similarity, 0 to 1; None when only the word search found it
    lexical_rank: int | None; lexical_score: float | None
    rerank_score: float | None   # set when a reranker ran

class Citation(BaseModel):
    marker: int; chunk_id: str; doc_id: str; title: str
    page: int | None             # the page that holds the cited text
    page_end: int | None         # set when the passage runs onto later pages
    section: str; snippet: str
    score: float                 # fused score (orders the sources)
    vector_score: float | None   # raw cosine, 0 to 1
    rerank_score: float | None   # when a reranker ran

class AnswerStatus(str, Enum):
    answered, no_answer, no_documents, provider_unavailable, retrieval_unavailable, rate_limited

class ProviderInfo(BaseModel): name: str; model: str | None; elapsed_ms: int

class Answer(BaseModel):         # the contract; every field always present
    id: str; question: str; collection_id: str
    status: AnswerStatus
    answer: str | None           # markdown with [n] markers, only when status == answered
    message: str | None          # one plain sentence when not answered
    citations: list[Citation]    # dense 1..K, in order of first citation, deduplicated by chunk
    closest: list[Citation]      # up to 3 nearest passages when status == no_answer, otherwise empty
    passages_considered: int
    provider: ProviderInfo | None
    knowledge_version: str
    language: str | None         # BCP-47 guess of the question's language
    created_at: datetime

JobStage = Literal["queued", "parsing", "chunking", "embedding", "writing", "done", "failed", "cancelled"]

class Job(BaseModel):
    id: str; collection_id: str; doc_id: str   # doc_id is empty until a new upload has created its document
    status: DocumentStatus       # queued while it waits for a worker, processing once a worker has started it
    error: str | None
    filename: str                # the file (or the path inside a folder or zip) this job ingests
    stage: JobStage
    chunks_done: int; chunks_total: int | None  # embedding progress
    bytes_total: int | None
    eta_seconds: int | None      # an estimate while the job runs
    cancel_requested: bool
    already_indexed: bool        # the same content is already ready in the collection: nothing new was stored
    replaced_doc_id: str | None; replaced_title: str | None   # the earlier version this upload replaces
    created_at: datetime; finished_at: datetime | None

class Feedback(BaseModel):
    id: str; answer_id: str; collection_id: str
    question: str                # copied from the answer, so a curator can read the vote without opening it
    rating: Literal["up", "down"]; note: str = ""; created_at: datetime

class ProviderStatus(BaseModel): name: str; installed: bool; signed_in: bool; version: str | None; detail: str; fix_command: str | None; models: list[str]
```

## 4. Interfaces

```python
# retrieval/base.py
class Embedder(Protocol):
    name: str; dims: int
    max_tokens: int              # the most tokens one passage can carry untruncated: the model's window minus
                                 # its special tokens and passage prefix (510 for bge-small-en-v1.5)
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def count_tokens(self, text: str) -> int: ...   # the model's own tokenizer; the chunker keeps every passage within max_tokens

class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...   # falls back to the input order on any error

def rrf(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]
def drop_near_duplicates(hits: list[Hit]) -> list[Hit]   # same normalized text, or an adjacent ordinal of the same doc whose text overlaps (one text's last 120 normalized characters appear in the other)

# store/base.py
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None
    def delete_doc(self, doc_id: str, *, keep_ordinals_below: int | None = None) -> int   # stale-tail cleanup when a bound is given
    def delete_collection(self, collection_id: str) -> int
    def vector_search(self, collection_id: str, vector: list[float], k: int, min_score: float = 0.0) -> list[Hit]
    def text_search(self, collection_id: str, query: str, k: int) -> list[Hit]
    def chunks_for(self, doc_id: str) -> list[Chunk]
    def neighbours(self, doc_id: str, ordinal: int, span: int = 1) -> list[Chunk]
    def count(self, collection_id: str | None = None) -> int
    def has_documents(self, collection_id: str) -> bool

class MetadataStore(Protocol):   # SQLite
    collections: create/get/list/delete
    documents: create/get/list(collection)/count_documents(collection) -> {status: n}/update/delete
               find_by_hash(collection, hash)/find_by_path(collection, source_path)/find_by_title(collection, title)
    jobs: create(collection, doc, filename=, bytes_total=)/get/list(collection, limit)
          update(job, status, error, doc_id=, **progress)   # stage, chunks_done, chunks_total, eta_seconds, already_indexed, replaced_doc_id, replaced_title
          request_cancel(job) -> Job | None; cancel_requested(job) -> bool
    answers and feedback: save_answer/get_answer; add_feedback/list_feedback(collection, limit)
    audit: audit(event, **fields); version: bump_version() -> str, current_version() -> str

# providers/base.py
class ProviderResult(BaseModel): text: str; model: str | None; elapsed_ms: int; raw: dict = {}
class Provider(Protocol):
    name: str
    def status(self) -> ProviderStatus
    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult   # raises ProviderError(kind) with kind in {not_installed, not_signed_in, timeout, rate_limited, failed}

# retrieval/service.py
class Retriever:
    def __init__(self, store, embedder, reranker=None, min_score=0.0, *, candidates=30, neighbour_expansion=True)
    @classmethod
    def from_settings(cls, store, embedder, settings, reranker=None) -> Retriever   # what the app, the MCP tools and the eval use
    def search(self, collection_id: str, query: str, top_k: int, *, budget_chars: int | None = None) -> list[Hit]
    # CANDIDATES by vector (floor applied) and CANDIDATES by full text; weighted RRF, with a passage that holds a
    # rare exact identifier or quoted phrase from the query boosted first; near-dup drop; the reranker orders
    # the pool. Without budget_chars (search): the first top_k. With it (an answer): at least top_k, more while
    # they fit and are not weak, then the neighbours of the top passages. Every hit is a RankedHit (relevance,
    # exact_match, neighbour_of) and serializes as a plain Hit.

# retrieval/reranker.py
def reranker_for(settings) -> LocalReranker | None   # LocalReranker(RERANK_MODEL) when RERANKER is local; .load() loads it now

# answer/compose.py
def compose(question, hits, provider, settings) -> Answer
```

## 5. Composition contract (`answer/prompts.py`, `answer/compose.py`)

System prompt (the generic persona is "a careful knowledge-base assistant"):

- Answer using ONLY the numbered passages.
- Write in the language of the question.
- Start with the direct answer in one or two sentences. Then give the useful details in short paragraphs or bullets. Use headings only for long answers.
- Cite inline with the passage number in square brackets, like [2], right after the sentence it supports. Do not write a sources list; the app does that.
- If the passages answer only part of the question, say which part is not covered in one final short paragraph.
- Never infer that something is absent or present because the passages do not mention it: say that the documents do not mention it, and stop.
- Do not say "based on the context" or similar.
- If the passages do not answer the question at all, reply with exactly `NO_ANSWER` and nothing else.

Composition rules:
- Budget: keep whole passages in rank order while they fit `MAX_CONTEXT_CHARS`; the first always fits. Only kept passages are numbered, so a dropped passage can never be cited.
- The sentinel check tolerates leading decoration (`*`, `_`, quotes, `#`, `>`) and case.
- `status`: `answered` when the model produced text; `no_answer` on the sentinel, or before any model call when a reranker scored the passages and its calibrated band calls every one weak; `no_documents` when the collection has no indexed documents (checked after an empty or declined result); `provider_unavailable` or `rate_limited` from provider errors, including a model the provider does not know (refused before any call, with a message that names the model and the ones to choose from); `retrieval_unavailable` when search fails.
- `message` texts: "The knowledge base does not cover this question." / "This collection has no documents yet. Upload some to get started." / "{provider} is not signed in. Run: {fix_command}" / "Search is unavailable right now."
- Citations: find `[n]` markers in the text, map to kept passages, renumber densely 1..K in order of first appearance, rewrite the text in one pass, dedupe by chunk. Unknown markers are left alone and logged. Snippet: the first 240 characters of the chunk's display text.
- Extractive provider: no model call. It returns a digest: a lead line ("Here is what the documents say about ...") then one bullet per source with the best sentences and a [n] marker, then the same citations. Its status is `answered` only when the top fused score is above a small threshold and the passages are about the question: each topic word of the question is weighted by its rarity across the retrieved passages (IDF), a word no passage contains is tolerated only once per five topic words, the passages together hold at least 60% of the question's weight and one passage alone at least 50%; otherwise `no_answer`. Its sentences are ranked by the same weights.

## 6. Provider adapters

Common: locate the binary (explicit setting, then PATH, then known install locations on Windows such as the VS Code extension folders), capture version, run with a timeout, send the prompt on **stdin**, never log the prompt, map exit codes and error texts to `ProviderError` kinds.

Claude Code (`providers/claude_code.py`):
- Command: `[bin, "-p", "--output-format", "json", "--no-session-persistence", "--model", model, "--system-prompt", system]`, prompt on stdin.
- Never pass `--bare` (it drops the stored sign-in).
- Parse JSON: `result` (text), `is_error`, `modelUsage` (keys are model ids), `duration_ms`, `session_id`. A result of "Not logged in" with `is_error` true means `not_signed_in`; the fix command is `claude /login` (or `claude` then `/login`).
- `status()`: run `[bin, "auth", "status"]`, parse JSON `loggedIn`, `subscriptionType`, `authMethod`.

Codex (`providers/codex.py`):
- Command: `[bin, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-C", tmpdir, "-o", outfile]` plus `-m model` when set and `-c model_reasoning_effort=<level>`; prompt on stdin; the system text is prepended to the prompt under an "Instructions" heading because Codex has no system-prompt flag.
- Read the answer from `outfile`; parse the token count from stdout when present.
- `status()`: run `[bin, "login", "status"]`; "Logged in using ChatGPT" means signed in; the fix command is `codex login`.
- Quota errors are mapped to `rate_limited` with the vendor's message.

Extractive (`providers/extractive.py`): always available; `generate` is not used; the composer calls its `digest(question, hits)`.

Registry (`providers/registry.py`): `statuses()` returns the statuses (cached briefly); `select(name_or_auto, model=None)` prefers `claude_code`, then `codex`, then `extractive`, and with a model the chosen provider does not know raises `ModelNotAvailableError` (a `ProviderError` whose plain message is shown as it is) before any call; the choice and the reason are logged. Every ask path (API, CLI) selects with the requested model.

## 7. HTTP API (prefix `/api`)

| Method and path | Purpose |
|---|---|
| `GET /api/health` | status, version, knowledge version, counts (`documents_ready`, `documents_failed`, `documents_processing`, `documents_total`), store health |
| `GET /api/providers` | `ProviderStatus[]` |
| `GET /api/collections`, `POST /api/collections` | list (each with `documents` = ready, `documents_total`, `documents_by_status`), create `{name, description}` |
| `DELETE /api/collections/{cid}` | delete with all documents; 200, or 202 with `pending` when it continues in the background |
| `GET /api/collections/{cid}/stats` | `documents_ready`, `documents_by_status`, chunks, by media type, last indexed |
| `POST /api/collections/{cid}/documents` | multipart: `files` (one or many), optional repeated `relative_paths` aligned with `files`, optional `title` (one file only); a `.zip` is unpacked into one job per file inside it; 202 with `{jobs, rejected, skipped}` (section 7.1) |
| `GET /api/collections/{cid}/documents` | list with status |
| `GET /api/documents/{did}` | one document |
| `GET /api/documents/{did}/chunks` | passages in ordinal order |
| `GET /api/documents/{did}/file` | the original file, always as a download |
| `POST /api/documents/{did}/reindex` | re-parse and re-embed from the kept original; 202 with a `Job` |
| `POST /api/documents/{did}/retry` | run a failed document again from the kept original; 202 with a `Job`; 409 for a document that did not fail (use reindex) |
| `DELETE /api/documents/{did}` | delete chunks, metadata and the original; 200, or 202 with `pending` when it continues in the background |
| `GET /api/jobs/{jid}`, `GET /api/collections/{cid}/jobs?limit=` | ingestion progress (`Job`, newest first) |
| `POST /api/jobs/{jid}/cancel` | ask a queued or running job to stop; returns the `Job` with `cancel_requested` |
| `GET /api/collections/{cid}/jobs/summary` | `{queued, processing, ready, failed, cancelled, replaced, duplicates, total}` |
| `POST /api/collections/{cid}/search` | `{query, top_k}` -> `Hit[]` (no model call) |
| `POST /api/collections/{cid}/ask` | `{question, top_k?, provider?, model?}` -> `Answer` |
| `GET /api/answers/{aid}` | a stored answer |
| `POST /api/answers/{aid}/feedback` | `{rating, note}` -> `Feedback` |
| `GET /api/collections/{cid}/feedback?limit=100` | the collection's `Feedback[]`, newest first |
| `GET /mcp` and `POST /mcp` | MCP streamable HTTP |
| `GET /` | the UI |

Error envelope: `{"error": {"code": "...", "message": "..."}}` with codes such as `not_found`, `unsupported_file_type`, `file_too_large`, `empty_document`, `encrypted_file`, `invalid_request`, `conflict`, `forbidden`, `unavailable`, `retrieval_unavailable`, `rate_limited`. Messages are plain sentences and never include paths or stack traces.

### 7.1 Rules behind the fields

**Host rules.** `/api` answers only requests addressed to this machine unless `RAGKB_API_PUBLIC=true`: a `Host` that is not `localhost`, `127.x.x.x` or `[::1]` (any port) gets 403 `forbidden`, except `GET /api/health`, which always answers for monitoring. A `POST`, `PUT`, `PATCH` or `DELETE` under `/api/` whose browser `Origin` is not the same host gets 403 as well, so another web page cannot upload, ask or delete through a local server. `/mcp` has its own guard, `RAGKB_MCP_PUBLIC` (section 9), so opening `/mcp` through a tunnel never opens `/api`. Both guards trust the `Host` header; a proxy that rewrites it to `localhost` defeats them.

**Document identity.** Within a collection a document is its `source_path`: the relative path of a file from a folder or zip upload, else the file name. Titles are labels only and may repeat.
- The same content (sha256) already ready in the collection is kept once: the job is `ready` at once with `already_indexed: true` and the existing `doc_id`; the summary counts it under `duplicates`.
- New content at an existing `source_path` replaces that document: the job carries `replaced_doc_id` and `replaced_title`, the old version stays searchable until the new one is ready, and a failed or cancelled replacement leaves the old version as it was. The summary counts it under `replaced`.
- Anything else is a new document, even when its title matches another one.

**Upload request and response.** `files` may be repeated; `relative_paths`, when sent, is repeated in the same order (the web page sends one file per request with its `webkitRelativePath`). Every file lands in exactly one of two lists of the 202 response `{"jobs": Job[], "rejected": [{"filename", "source_path", "code", "message"}], "skipped": n}`:
- `jobs`: one per accepted file. A file that is accepted and then fails to index stays a job (and a document) that ends `failed` with the reason.
- `rejected`: one per file refused up front by validation (an unsupported type, too large, empty, password protected: `encrypted_file`, or a path outside the upload). A refused file leaves no job and no document row.
- `skipped`: the zip entries left out on purpose (folders, hidden files, archive metadata).

A `.zip` is unpacked on the server into one job per file inside it, each identified by its path inside the archive (`source_path`). The rules apply per file: a file inside the archive that is refused gets its own entry in `rejected`, and the rest of the archive still goes in. The archive is refused as a whole only when it is empty, cannot be opened, holds more than 10,000 entries, or is (or would expand to) more than ten times the per-file limit. When nothing in the request is accepted, the answer is the error envelope with the first rejection's status and code; an error for the whole request (for example a malformed form) also uses the envelope.

**Job progress and cancel.** A job moves through the stages `queued`, `parsing`, `chunking`, `embedding`, `writing` and ends `done`, `failed` or `cancelled`; `status` is `queued` until a worker starts it and `processing` from then on, then `ready`, `failed` or `cancelled`. While embedding, `chunks_done` of `chunks_total` advance batch by batch (`EMBED_BATCH_SIZE` passages) and `eta_seconds` estimates the rest. `POST /api/jobs/{jid}/cancel` sets `cancel_requested`: a waiting job ends `cancelled` at once; a running one stops at the next stage or batch boundary (writing, once started, is not interrupted) and ends `cancelled`. A cancelled new upload leaves no document; a cancelled replacement, re-index or retry leaves the earlier version as it was. `GET /api/collections/{cid}/jobs` lists jobs newest first, up to `limit` (at most 500).

**Deletes.** Deleting a document or a collection first cancels its waiting and running jobs, then runs apart from the ingestion workers. The request waits up to 15 seconds: 200 when the delete finished, or 202 with `{"pending": true, "message"}` when it goes on in the background and finishes there.

**Counts.** The headline count everywhere is **searchable documents** (status `ready`): the collections list `documents`, the stats `documents_ready`, the health `documents_ready`, and the MCP `list_collections` count. Documents still processing, failed or cancelled are reported next to it (`documents_by_status`, `documents_total`, `documents_failed`, `documents_processing`) and shown as a secondary note, never added to the headline.

**Page spans.** A chunk records the first page it starts on (`page`) and every page it touches (`page_start` to `page_end`). A citation's `page` is the page that holds the cited text, and `page_end` is set only when the passage runs onto a later page; the UI shows "pages 3 to 4".

**Scores.** `score` is the fused rank score: it orders hits and citations and is not a measure of relevance (typical values 0.01 to 0.04). `vector_score` is the raw cosine similarity between the question and the passage, 0 to 1, on one scale for every query; it is `null` when only the word search found the passage. `rerank_score` is the cross-encoder's score when `RAGKB_RERANKER=local` ran. The UI draws its bar from `vector_score`, never relative to the other sources.

**Closest passages.** When `status` is `no_answer`, `closest` holds up to three of the best passages found, with the same fields as citations, so a person can judge the decline for themselves. They are never cited and never turned into answer text: `answer` stays `null` (the honesty contract). For every other status `closest` is empty.

**Feedback read-back.** `POST /api/answers/{aid}/feedback` stores the vote with its `collection_id` and a copy of the question. `GET /api/collections/{cid}/feedback?limit=100` returns the collection's votes newest first, so a curator can review them and open each answer by its `answer_id`.

## 8. UI (`ragkb/ui`)

Single page, three regions: a left sidebar (collections, health badge, provider badge, theme switch), a main area with tabs (Documents, Ask, Settings, Connect, Help), and a status bar (knowledge version, data folder). Plain HTML, CSS and JavaScript with `fetch`; markdown rendering for answers with a small safe renderer (escape everything, then format headings, bold, lists and `[n]` markers); no external resources and no inline script (the page's CSP forbids both).

Addresses: `#/c/<collection id>/<tab>`, `#/<tab>` when there is no collection, and `#/answer/<answer id>` for a stored answer. Selecting a tab or a collection adds a history entry, so Back and Forward move between them, and a reload opens the same view. The older `#<tab>` and `#ask/<answer id>` forms still open and are rewritten.

States to render distinctly: empty collection; uploading with byte progress; per-file job progress (stage, passages done of total, time left) with Cancel while queued or running; processing; failed with reason, Retry and Dismiss; replaced, already indexed and rejected uploads; a pending state with a spinner, and a note after two seconds, for delete, dismiss, re-index and retry; answer; no answer with the closest passages; no documents; provider not signed in (with the fix command and a copy button); rate limited; search unavailable.

Layout and data rules:
- The documents table never grows wider than its container: fixed layout, fixed status and action columns, titles and paths wrap, and as the container narrows the Uploaded, Pages and Indexed columns move into the line under the title. The passages drawer spans every visible column.
- Counts follow section 7.1: searchable documents first, the others as a note.
- The last known health and provider states are kept in `localStorage` and shown at once on a reload until fresh ones arrive.
- Upload responses are read in their documented shape (section 7.1): a row per entry of `jobs` and `rejected`, and the `skipped` count on a zip's row. Retry falls back to re-index only on its 409 (a row that did not fail), and a delete's 202 `pending` is announced with the server's message.
- The Raw response block offers the same request for bash, PowerShell and cmd, remembered per browser. The JSON body is written with every character other than letters, digits and plain punctuation as a `\uXXXX` escape, so the command is plain ASCII and needs no shell-specific quoting beyond the outer quotes.
- Settings has a Feedback panel with the collection's votes and a link to each answer.

## 9. MCP tools (`ragkb/mcp/server.py`)

- `list_collections()` -> `[{id, name, documents}]` (`documents` counts searchable documents)
- `list_documents(collection_id)` -> `[{id, title, pages, chunks, indexed_at}]`
- `search_knowledge_base(collection_id, query, top_k=6)` -> passages with title, page, section, text, score
- `ask_knowledge_base(collection_id, question, top_k=6)` -> the `Answer` contract (with `closest` and the score fields) produced with the **extractive** provider from the same retrieval as an answer in the app (`budget_chars=MAX_CONTEXT_CHARS`), plus `guidance`: a short instruction for the calling assistant on how to write the final answer from the brief. (The assistant is the user's own model; we never call a model here.)

Loopback only unless `RAGKB_MCP_PUBLIC=true`.

## 10. Testing conventions

- Offline by default: `FakeEmbedder` (deterministic hash-based vectors), the extractive provider, a temporary `DATA_DIR`, and no reranker: an autouse fixture sets `RAGKB_RERANKER=none`, and an offline test that still tries to load the real cross-encoder fails (a test that needs the model is live; one that stands in for it patches `TextCrossEncoder`). The offline suite downloads nothing.
- Provider adapters are tested against fake CLI scripts placed on PATH that emit recorded outputs, covering success, not signed in, timeout and quota error.
- Live tests run only with `RAGKB_LIVE_TESTS=1` and use the real embedder and the installed CLIs.
- Hard gates: tenancy (a collection never sees another collection's chunks), honesty (a declined answer has `answer: null`), citation integrity (every marker maps to a kept passage), re-upload durability (the old version survives a failed ingest).
