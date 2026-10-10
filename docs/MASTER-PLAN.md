# rag-kb: master plan

**What it is.** An enterprise knowledge base with retrieval-augmented answers. People upload a document corpus, the system vectorizes it locally, and anyone can ask questions and get a readable, well-organized, cited answer. The AI that writes the answer runs on **each user's own Claude or ChatGPT subscription**. There are no cloud AI keys, no AWS and no shared API bill.

**Two objectives, in order.**
1. **Run it locally.** A repo anyone can clone and run on their own machine in a few minutes, with their own subscription.
2. **Run it as a web product.** One deployment, many users, each bringing their own subscription.

This plan builds objective 1 completely and designs objective 2 so that nothing in objective 1 has to be thrown away.

---

## 1. Hard requirements

| Requirement | How it is met |
|---|---|
| No AI API keys, no AWS | Embeddings and reranking run locally on CPU with open models (fastembed, ONNX). Answer generation goes through the user's own subscription (section 2). |
| Each user's own Claude or ChatGPT subscription powers the answers | Locally: the app drives the official command-line tools that those subscriptions include, Claude Code (`claude -p`) and OpenAI Codex (`codex exec`), both signed in once by the user. On the web: the user's own claude.ai or ChatGPT connects to the knowledge base as a tool through MCP, so their assistant writes the answer (section 2). |
| Upload a whole corpus through a UI | Drag-and-drop upload of many files, with per-document status, re-index and delete. PDF, Word, Markdown, text, HTML, CSV and Excel. |
| Very good RAG | Structure-aware chunking that keeps section and page metadata, hybrid retrieval (vector plus full-text) with rank fusion, near-duplicate suppression, an optional local reranker, and an evaluation harness that measures recall, citation quality and false answers. |
| Answers are readable and user friendly, not citation dumps | A composition step turns the retrieved passages into a direct answer first, then the details, in the user's language, with inline [n] citations that map to a clean sources list. When the documents do not answer, it says so plainly instead of guessing. |
| Enterprise shape | Collections (spaces), document management, content-hash dedupe, a knowledge version on every answer, audit of ingestion and answers, feedback on answers, honest degraded states, and a path to multi-user auth and roles in objective 2. |

## 2. How "your own subscription" works

This is the part that shapes everything else, so it is spelled out.

### 2.1 Locally (objective 1)

Both subscription vendors ship an official command-line agent that signs in with the consumer subscription and can run non-interactively:

| Subscription | Tool | How we call it | Verified on this machine |
|---|---|---|---|
| Claude Pro or Max | **Claude Code** (`claude`) | `claude -p --output-format json --no-session-persistence --model <model> --system-prompt <text>` with the prompt on stdin | Yes, 10 October 2026: Claude Code 2.1.292 signed in through claude.ai with a Max subscription answered a headless prompt in 12 s on Sonnet 5. The `--bare` flag must not be used, because it skips the stored login. |
| ChatGPT Plus or Pro | **Codex CLI** (`codex`) | `codex exec --skip-git-repo-check --ephemeral -s read-only -C <tmp> -o <file>` with the prompt on stdin | Yes, 10 October 2026: Codex 0.162 "Logged in using ChatGPT" answered in 6 s. |

The app detects which tools are installed and signed in, lets the user pick, and shows the exact sign-in command when one is missing. There is also a **no-AI mode** (extractive) that returns a structured, readable digest of the best passages, so the knowledge base works even with no subscription at all.

What this means for the user: install the app, sign in to Claude Code and/or Codex once (as they already do to use those tools), and every answer runs on their plan. The app never sees a password or a token; it only runs the vendor's tool.

### 2.2 On the web (objective 2)

A hosted server cannot run a user's Claude Code or Codex, because those sign-ins live on the user's machine. The web product therefore turns the model around: the knowledge base becomes a **tool the user's own assistant calls**.

- The server exposes the knowledge base as a remote **MCP server** with a sign-in flow (OAuth). Both claude.ai (Pro, Max, Team) and ChatGPT (Plus and Pro, through connectors) let a user add a custom connector with their own account.
- The user asks their question in claude.ai or ChatGPT. Their assistant calls `ask_knowledge_base`, receives a well-structured brief (the best passages, a structured extractive digest, the honesty flags and a sources list, plus formatting guidance), and writes the readable answer. The generation runs on their subscription.
- Our web UI handles everything else: uploading and managing the corpus, collections and roles, the connector link and setup steps, usage and feedback.

This is the same split as objective 1 (retrieval and honesty are ours, generation is theirs), so the retrieval core, the answer contract, the MCP tools and the UI carry over unchanged. Objective 1 already ships the MCP server for local use (Claude Desktop and claude.ai can connect to a local or tunneled instance), which is the seed of objective 2.

### 2.3 What we will not do

- No scraping of consumer sessions, no reuse of another product's login tokens, no spoofing of a vendor's client. The two CLIs and the connector flow are the vendors' own supported surfaces for subscription use.
- No API keys as a requirement. An optional "bring your own OpenAI-compatible endpoint" provider (for example a local open model) may be added later for offline or air-gapped use, but it is not part of the product promise.

## 3. Objective 1: scope

### 3.1 User-facing

- **Collections.** Create, switch, delete. One knowledge base can hold several corpora (teams, topics, clients).
- **Documents.** Drag-and-drop or pick many files. Each shows its status (queued, processing, ready, failed with reason), title, type, pages, chunks, size, uploaded and indexed times. Actions: open the original, view its passages, re-index, delete (two-click, with a short arm time). Re-uploading the same content is detected by hash and reported as "already indexed"; the same title with new content replaces the old version.
- **Ask.** A question box with provider and model choice, passage count, and the answer rendered as formatted text. Inline [n] markers highlight the matching source card. Sources show title, page or section, a snippet and a relevance bar. Thumbs up or down with an optional note. Elapsed time and model shown. Clear, distinct states for "no documents yet", "the documents do not cover this", "provider not signed in" and "search unavailable".
- **Settings and health.** Which providers are installed and signed in, with the command to fix each; the embedding model; index size; the data folder; the knowledge version.
- **Connect your assistant.** The local MCP address and setup steps for Claude Desktop and claude.ai, with the tool list.
- **Help.** How answers are produced, how to read scores, what the states mean.

### 3.2 Under the hood

- **Ingestion.** Parsers per format keep structure: headings, page numbers, tables as blocks. Chunking packs blocks into about 350 tokens with overlap, never splits a short block, keeps the heading path and page on every chunk, and prefixes each chunk with its title and section for embedding (the display text stays clean). Deterministic chunk ids, content-hash dedupe, embed everything first, write in one transaction, then drop the stale tail of a replaced document. A knowledge version `v{YYYYMMDD}.{n}` bumps on every change and is stamped on chunks and answers. Raw files are kept for download and re-index.
- **Retrieval.** Local embeddings (default `BAAI/bge-small-en-v1.5`, 384 dimensions, with a multilingual option), LanceDB as the embedded vector and full-text store (no server to run), rank fusion of the two result lists, near-duplicate suppression, an optional local cross-encoder reranker, a configurable minimum score that applies to vector scores only.
- **Answer composition.** One prompt, one contract: answer in the question's language; lead with the direct answer; then the details; cite inline with [n]; add "what the documents do not say" when the question is only partly covered; reply with an exact sentinel when the passages do not answer, which the app turns into an honest "not covered" state rather than text the user might mistake for an answer. Citations are renumbered densely and deduplicated by source. Every response has every field, always.
- **Providers.** `claude_code`, `codex`, `extractive`, behind one interface, with detection, sign-in status, timeouts and clear errors. Prompts travel on stdin, so long contexts are safe on Windows.
- **MCP server.** `search_knowledge_base`, `ask_knowledge_base`, `list_documents`, `list_collections` over streamable HTTP at `/mcp`, loopback only by default.
- **Evaluation.** A golden-set format (questions with expected sources, plus negatives that must be declined), scorers for recall@k, MRR, citation rate and false-answer rate, a report, and a sample golden set over the sample corpus. The offline run uses a deterministic fake embedder and the extractive provider so it works in CI; the live run uses the real models and a chosen provider.
- **Operations.** Structured logs, an audit trail of uploads, deletes and answers (question, provider, model, elapsed, status, knowledge version; never the full passages), per-process rate limits on expensive calls, and a `doctor` command.

### 3.3 Out of scope for objective 1

OCR of scanned PDFs, multi-user auth and roles, cloud storage, scheduled re-crawls of external sources, streaming answers in the UI (the contract allows adding it), and an OpenAI-compatible provider.

## 4. Objective 2: scope (designed now, built later)

- **Users and tenancy.** Sign-in (OIDC or email link), organizations, collections with roles (owner, editor, reader), server-side role resolution only, audit per user.
- **Storage.** Postgres with pgvector (or LanceDB on a volume for small deployments) behind the same store interface; object storage for originals; migrations.
- **Connector sign-in.** OAuth for the MCP server with client registration, an allow-list of redirect hosts, hashed tokens, a session cap, revoke by user, and a sign-in page that names the connecting app.
- **Answer brief for assistants.** `ask_knowledge_base` returns the extractive digest, passages, flags and formatting guidance so claude.ai or ChatGPT produce the final readable answer on the user's subscription. The web UI shows the same brief so users can check what their assistant saw.
- **Operations.** Docker image and compose, health and readiness, metrics and alarms, backups, rate limits in Redis, a deploy runbook.
- **Optional later.** Streaming, OCR, scheduled ingestion from shared drives, a local-bridge mode where the web UI chats through the user's own CLI on their machine.

## 5. Stack decisions

| Area | Choice | Why |
|---|---|---|
| Language and API | Python 3.11+, FastAPI | Fast to build, typed, async-friendly, familiar. |
| Vector and full-text store | **LanceDB** (embedded) | No server to install, one `pip install`, vector plus BM25 full-text plus hybrid in one library, fast on a laptop, portable data folder. Behind a store interface so pgvector can replace it for objective 2. |
| Metadata | SQLite (standard library) | Collections, documents, jobs, feedback and audit need transactions and status updates; zero setup. |
| Embeddings | **fastembed** with `BAAI/bge-small-en-v1.5` default | ONNX on CPU, no PyTorch, about 130 MB, good English quality. `intfloat/multilingual-e5-small` as the documented option for mixed-language corpora. Verified on this machine: 5 s to load and embed. |
| Reranker | fastembed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`), off by default | Local, cheap, measurable gain; off until the eval shows it helps. |
| Parsing | pypdf, python-docx, markdown-it-py, BeautifulSoup, openpyxl, csv | Light, pure Python, cover the formats people actually upload. |
| Answer generation | Claude Code CLI, Codex CLI, extractive | Section 2. |
| UI | Single-page app in plain HTML, CSS and JavaScript served by FastAPI, no build step | Running locally must not require Node or a bundler. |
| Packaging | `pip install -e .` and a `ragkb` command; Docker as an option | One command to run. |
| Tests | pytest, offline by default; live tests opt-in by environment variable | CI without subscriptions. |

## 6. Quality bar

- **Retrieval:** recall@5 at least 0.85 on the sample golden set; false-answer rate on negatives at most 0.05.
- **Answers:** every answered question cites at least one passage; no citation points at a passage the budget dropped; every "not covered" state is a status, never free text that reads like an answer; answers come back in the language of the question.
- **Readability:** direct answer first, then details; no "based on the context" preambles; sources list deduplicated and in citation order.
- **Reliability:** an ingestion failure leaves the previous version of a document intact; a provider failure returns a clear status; a store failure never crashes the app.
- **Speed:** search under 300 ms on a few thousand chunks; a full answer within the provider's time (10 to 30 s typical).

## 7. Security and privacy

- Documents never leave the machine except as passages inside the prompt sent to the user's own subscription provider, which is the same trust boundary as pasting them into that assistant.
- No secrets are stored by the app; the vendor tools hold their own logins.
- The MCP server binds to loopback by default; exposing it is a deliberate flag, and objective 2 adds OAuth before any public exposure.
- Uploads are size-capped and type-checked; HTML is parsed to text; nothing uploaded is executed.
- The audit log records what was asked, by which provider and with what outcome, never the passages or the full answer.

## 8. Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Plan, contracts, skeleton, repo | This document, `docs/ARCHITECTURE.md`, the package skeleton pushed to GitHub |
| 1 | Ingestion, store, retrieval | Sample corpus indexes; search returns the right passages; offline tests pass |
| 2 | Providers and composition | `ragkb ask` gives a cited, readable answer through Claude Code and through Codex; declines honestly; extractive mode works |
| 3 | API and UI | Upload, manage, ask, feedback, settings in the browser; states render correctly |
| 4 | MCP, evals, docs | Claude Desktop connects locally; `ragkb eval` reports; README quickstart works on a clean machine |
| 5 | Hardening | Integration run end to end, review fixes, Windows and macOS paths checked, tagged `v0.1.0` |
| 6+ | Objective 2 | Per section 4, starting with auth and the connector sign-in |

## 9. Risks

| Risk | Mitigation |
|---|---|
| A vendor changes its CLI flags or output | Providers are isolated adapters with version checks and tests against recorded outputs; `ragkb doctor` reports the installed version. |
| Codex quota on some plans is small | Default to Claude Code when both are present; show the quota error plainly; extractive mode as a fallback. |
| Embedding quality on non-English corpora | The multilingual model option and the eval harness to prove it. |
| Chunking that breaks tables or definitions | Block-aware packing with tests on tricky documents. |
| LanceDB on Windows | Verified on this machine; the store interface allows a swap. |
| People mistake a decline for an answer | Status-driven UI states and the sentinel contract. |

## 10. Open decisions

- **License** for the repo (private today). MIT if it is to be shared freely, otherwise a proprietary notice.
- **Default embedding model**: English-first small model, or multilingual from day one at some cost in speed.
- **Reranker on by default** once the eval shows the gain.
- **OCR** for scanned PDFs (objective 2 candidate).
