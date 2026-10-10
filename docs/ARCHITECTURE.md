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
    chunker.py         blocks -> chunks (about 350 tokens, overlap, metadata kept)
    pipeline.py        upload -> parse -> chunk -> embed -> store, status and version
  store/
    base.py            VectorStore protocol, MetadataStore protocol
    lancedb_store.py   chunks and vectors, full-text index, hybrid search
    sqlite_meta.py     collections, documents, jobs, feedback, audit, version counter
  retrieval/
    base.py            Embedder, Reranker protocols; Hit; RRF; near-duplicate drop
    embedder.py        fastembed embedder; FakeEmbedder for tests
    reranker.py        fastembed cross-encoder reranker; NoReranker
    service.py         Retriever: query -> ranked hits
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
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `RERANKER` | `none` | `none` or `local` |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed reranker |
| `TOP_K` | `6` | passages given to the composer |
| `MIN_SCORE` | `0.0` | vector-score floor, 0 to 1; lexical hits bypass it |
| `MAX_CONTEXT_CHARS` | `12000` | composition budget |
| `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS` | `350`, `60` | chunking targets (tokens approximated as words times 1.3) |
| `MAX_UPLOAD_MB` | `50` | per file |
| `PROVIDER` | `auto` | `auto`, `claude_code`, `codex`, `extractive` |
| `CLAUDE_BIN`, `CODEX_BIN` | auto-detected | explicit paths to the CLIs |
| `CLAUDE_MODEL` | `sonnet` | passed to `--model` |
| `CODEX_MODEL` | empty (the CLI's own default) | passed to `-m` when set |
| `CODEX_REASONING` | `medium` | passed as `-c model_reasoning_effort=...` |
| `ANSWER_TIMEOUT_S` | `180` | provider call timeout |
| `MCP_ENABLED` | `true` | mount `/mcp` |
| `UI_ENABLED` | `true` | mount the UI at `/` |
| `LOG_LEVEL` | `INFO` | |

A test asserts that every setting read in code appears in `.env.example` and the reverse.

## 3. Data model (`ragkb/models.py`)

```python
class Collection(BaseModel):
    id: str; name: str; description: str = ""; created_at: datetime

class DocumentStatus(str, Enum): queued, processing, ready, failed

class Document(BaseModel):
    id: str                      # uuid4 hex
    collection_id: str
    title: str
    filename: str
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
    page: int | None; char_start: int; char_end: int; token_estimate: int
    knowledge_version: str; created_at: datetime

class Hit(BaseModel):
    chunk: Chunk; score: float; vector_score: float | None; lexical_rank: int | None; rerank_score: float | None

class Citation(BaseModel):
    marker: int; chunk_id: str; doc_id: str; title: str; page: int | None; section: str; snippet: str; score: float

class AnswerStatus(str, Enum):
    answered, no_answer, no_documents, provider_unavailable, retrieval_unavailable, rate_limited

class ProviderInfo(BaseModel): name: str; model: str | None; elapsed_ms: int

class Answer(BaseModel):         # the contract; every field always present
    id: str; question: str; collection_id: str
    status: AnswerStatus
    answer: str | None           # markdown with [n] markers, only when status == answered
    message: str | None          # one plain sentence when not answered
    citations: list[Citation]    # dense 1..K, in order of first citation, deduplicated by chunk
    passages_considered: int
    provider: ProviderInfo | None
    knowledge_version: str
    language: str | None         # BCP-47 guess of the question's language
    created_at: datetime

class Job(BaseModel): id: str; collection_id: str; doc_id: str; status: DocumentStatus; error: str | None; created_at: datetime; finished_at: datetime | None
class Feedback(BaseModel): answer_id: str; rating: Literal["up", "down"]; note: str = ""; created_at: datetime
class ProviderStatus(BaseModel): name: str; installed: bool; signed_in: bool; version: str | None; detail: str; fix_command: str | None; models: list[str]
```

## 4. Interfaces

```python
# retrieval/base.py
class Embedder(Protocol):
    name: str; dims: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...   # falls back to the input order on any error

def rrf(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]
def drop_near_duplicates(hits: list[Hit]) -> list[Hit]   # same normalized text, or adjacent ordinal of the same doc

# store/base.py
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None
    def delete_doc(self, doc_id: str, *, keep_ordinals_below: int | None = None) -> int   # stale-tail cleanup when a bound is given
    def delete_collection(self, collection_id: str) -> int
    def vector_search(self, collection_id: str, vector: list[float], k: int, min_score: float = 0.0) -> list[Hit]
    def text_search(self, collection_id: str, query: str, k: int) -> list[Hit]
    def chunks_for(self, doc_id: str) -> list[Chunk]
    def count(self, collection_id: str | None = None) -> int
    def has_documents(self, collection_id: str) -> bool

class MetadataStore(Protocol):   # SQLite
    collections: create/get/list/delete; documents: create/get/list(collection)/update_status/delete/find_by_hash(collection, hash)/find_by_title(collection, title)
    jobs: create/get/update; feedback: add/list; audit: add(event, **fields); version: bump() -> str, current() -> str

# providers/base.py
class ProviderResult(BaseModel): text: str; model: str | None; elapsed_ms: int; raw: dict = {}
class Provider(Protocol):
    name: str
    def status(self) -> ProviderStatus
    def generate(self, system: str, prompt: str, *, timeout_s: int, model: str | None = None) -> ProviderResult   # raises ProviderError(kind) with kind in {not_installed, not_signed_in, timeout, rate_limited, failed}

# retrieval/service.py
class Retriever:
    def search(self, collection_id: str, query: str, top_k: int) -> list[Hit]
    # vector top_k*4 (floor applied), lexical top_k*4, RRF to top_k*3, near-dup drop, optional rerank, slice top_k

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
- Do not say "based on the context" or similar.
- If the passages do not answer the question at all, reply with exactly `NO_ANSWER` and nothing else.

Composition rules:
- Budget: keep whole passages in rank order while they fit `MAX_CONTEXT_CHARS`; the first always fits. Only kept passages are numbered, so a dropped passage can never be cited.
- The sentinel check tolerates leading decoration (`*`, `_`, quotes, `#`, `>`) and case.
- `status`: `answered` when the model produced text; `no_answer` on the sentinel; `no_documents` when the collection has no indexed documents (checked after an empty or declined result); `provider_unavailable` or `rate_limited` from provider errors; `retrieval_unavailable` when search fails.
- `message` texts: "The knowledge base does not cover this question." / "This collection has no documents yet. Upload some to get started." / "{provider} is not signed in. Run: {fix_command}" / "Search is unavailable right now."
- Citations: find `[n]` markers in the text, map to kept passages, renumber densely 1..K in order of first appearance, rewrite the text in one pass, dedupe by chunk. Unknown markers are left alone and logged. Snippet: the first 240 characters of the chunk's display text.
- Extractive provider: no model call. It returns a digest: a lead line ("Here is what the documents say about ...") then one bullet per source with the best sentences and a [n] marker, then the same citations. Its status is `answered` only when the top fused score is above a small threshold; otherwise `no_answer`.

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

Registry (`providers/registry.py`): `available()` returns statuses; `select(name_or_auto)` prefers `claude_code`, then `codex`, then `extractive`; the choice and the reason are logged.

## 7. HTTP API (prefix `/api`)

| Method and path | Purpose |
|---|---|
| `GET /api/health` | status, version, knowledge version, counts, store health |
| `GET /api/providers` | `ProviderStatus[]` |
| `GET /api/collections`, `POST /api/collections` | list, create `{name, description}` |
| `DELETE /api/collections/{cid}` | delete with all documents |
| `GET /api/collections/{cid}/stats` | documents, chunks, by media type, last indexed |
| `POST /api/collections/{cid}/documents` | multipart, one or many files, optional `title`; returns jobs; dedupe by hash; same title replaces |
| `GET /api/collections/{cid}/documents` | list with status |
| `GET /api/documents/{did}` | one document |
| `GET /api/documents/{did}/chunks` | passages in ordinal order |
| `GET /api/documents/{did}/file` | the original file |
| `POST /api/documents/{did}/reindex` | re-parse and re-embed from the kept original |
| `DELETE /api/documents/{did}` | delete chunks, metadata and the original |
| `GET /api/jobs/{jid}`, `GET /api/collections/{cid}/jobs` | ingestion progress |
| `POST /api/collections/{cid}/search` | `{query, top_k}` -> `Hit[]` (no model call) |
| `POST /api/collections/{cid}/ask` | `{question, top_k?, provider?, model?}` -> `Answer` |
| `GET /api/answers/{aid}` | a stored answer |
| `POST /api/answers/{aid}/feedback` | `{rating, note}` |
| `GET /mcp` and `POST /mcp` | MCP streamable HTTP |
| `GET /` | the UI |

Error envelope: `{"error": {"code": "...", "message": "..."}}` with codes such as `not_found`, `unsupported_file_type`, `file_too_large`, `empty_document`, `invalid_request`, `provider_unavailable`, `rate_limited`. Messages are plain sentences and never include paths or stack traces.

## 8. UI (`ragkb/ui`)

Single page, three regions: a left sidebar (collections, health badge, provider badge), a main area with tabs (Documents, Ask, Settings, Connect, Help), and a status bar (knowledge version, data folder). Plain HTML, CSS and JavaScript with `fetch`; markdown rendering for answers with a small safe renderer (escape everything, then format headings, bold, lists and `[n]` markers); no external CDN required to function.

States to render distinctly: empty collection, uploading with progress and per-file status, processing, failed with reason, answer, no answer, no documents, provider not signed in (with the fix command and a copy button), rate limited, search unavailable.

## 9. MCP tools (`ragkb/mcp/server.py`)

- `list_collections()` -> `[{id, name, documents}]`
- `list_documents(collection_id)` -> `[{id, title, pages, chunks, indexed_at}]`
- `search_knowledge_base(collection_id, query, top_k=6)` -> passages with title, page, section, text, score
- `ask_knowledge_base(collection_id, question, top_k=6)` -> the `Answer` contract produced with the **extractive** provider plus `guidance`: a short instruction for the calling assistant on how to write the final answer from the brief. (The assistant is the user's own model; we never call a model here.)

Loopback only unless `RAGKB_MCP_PUBLIC=true`.

## 10. Testing conventions

- Offline by default: `FakeEmbedder` (deterministic hash-based vectors), the extractive provider, a temporary `DATA_DIR`.
- Provider adapters are tested against fake CLI scripts placed on PATH that emit recorded outputs, covering success, not signed in, timeout and quota error.
- Live tests run only with `RAGKB_LIVE_TESTS=1` and use the real embedder and the installed CLIs.
- Hard gates: tenancy (a collection never sees another collection's chunks), honesty (a declined answer has `answer: null`), citation integrity (every marker maps to a kept passage), re-upload durability (the old version survives a failed ingest).
