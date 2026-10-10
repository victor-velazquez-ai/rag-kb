# rag-kb user guide

rag-kb turns a folder of documents into a knowledge base you can question. You upload the documents, rag-kb splits them into passages and indexes them on your machine, and every question gets a direct, readable answer with numbered citations back to the passages it came from. The answer is written by your own Claude or ChatGPT subscription; when the documents do not answer the question, rag-kb says so instead of guessing.

## 1. What you need

- **Python 3.11 or newer** (`python --version`).
- About **1 GB of free disk**: the embedding model is about 130 MB, the rest is your documents and their index.
- For AI-written answers, one of:
  - a **Claude Pro or Max** plan and Claude Code, or
  - a **ChatGPT Plus or Pro** plan and Codex.

  Without either, rag-kb still works in **extractive mode**: answers are a digest of the most relevant sentences, with citations, and no AI.

## 2. Install

```bash
git clone https://github.com/victor-velazquez-ai/rag-kb.git
cd rag-kb
python -m venv .venv
```

Activate the environment, then install:

```powershell
# Windows PowerShell
.venv\Scripts\activate
pip install -e .
```
```bash
# macOS / Linux
source .venv/bin/activate
pip install -e .
```

Use `pip install -e ".[dev]"` instead if you also want to run the tests (`pytest`).

## 3. Sign in to Claude Code or Codex

You only do this once, and only for the tool that matches your subscription. Details, and what rag-kb runs on your behalf, are in [PROVIDERS.md](PROVIDERS.md).

| Subscription | Install | Sign in | Check |
|---|---|---|---|
| Claude Pro or Max | `irm https://claude.ai/install.ps1 \| iex` (Windows PowerShell) or `curl -fsSL https://claude.ai/install.sh \| bash` (macOS, Linux) | `claude auth login` | `claude auth status` shows `"loggedIn": true` |
| ChatGPT Plus or Pro | `npm install -g @openai/codex` | `codex login`, then "Sign in with ChatGPT" | `codex login status` prints `Logged in using ChatGPT` |

If you already use the Claude Code or Codex extension in VS Code, rag-kb finds the program inside the extension, so only the sign-in is needed.

## 4. Check and start

```bash
ragkb doctor
```

`doctor` checks Python, the data folder, the stores, the embedding model (the first run downloads it), the answer providers and the MCP server. Every line should read `[  ok]`. A provider that is not ready shows the exact command that fixes it.

```bash
ragkb serve --open
```

This starts rag-kb at **http://127.0.0.1:8765** and opens your browser. Leave the terminal open; Ctrl+C stops the server. `--port 9000` picks another port.

Start rag-kb from the same folder each time (or set `RAGKB_DATA_DIR` to an absolute path, see section 9): the data folder is `./data` relative to where you start it.

## 5. Create a collection and upload

A **collection** holds one corpus: a team's handbook, a product's manuals, a client's contracts. Questions are answered from one collection at a time, and collections never see each other's documents.

1. In the left sidebar, under **New collection**, type a name (and optionally a description) and choose **Create collection**.
2. On the **Documents** tab, drop files on the drop zone or choose **choose files**. You can add many at once.
   - Types: PDF, Word (`.docx`), Markdown, text, HTML, CSV and TSV, Excel (`.xlsx`). Older `.doc` and `.xls` files must be saved as `.docx` or `.xlsx` first.
   - Size: up to 50 MB per file (`RAGKB_MAX_UPLOAD_MB`).
   - **Title** (optional, one file only): the name shown in citations. Otherwise the document's own title or the file name is used.
3. Each file shows its progress under **Uploads**, then appears in the documents table with its status:

| Status | Meaning |
|---|---|
| queued | Waiting its turn. |
| processing | Being read, split into passages and embedded. |
| ready | Indexed and searchable. |
| failed | Not indexed; the reason is shown (for example "This PDF has no extractable text": scanned PDFs need OCR, which this version does not do). |

Uploading the **same content again** is detected and reported as "Already indexed"; nothing is duplicated. Uploading a document with the **same title but new content** replaces the old version. The old version stays searchable until the new one is fully indexed, so a failed upload never loses the document you had.

From the command line, the same thing:

```bash
ragkb ingest path/to/folder another-file.pdf --collection handbook
```

Folders are read recursively; the collection is created when it does not exist.

## 6. Ask

Open the **Ask** tab, type a question and press Enter (Shift+Enter adds a line; `/` anywhere jumps back to the question box).

- **Provider**: Auto (Claude Code if signed in, then Codex, then extractive), or pick one for this question.
- **Model**: for example `sonnet`, `opus` or `haiku` for Claude Code; Codex offers the models your install knows.
- **Passages**: how many passages are given to the provider (default 6). More passages help broad questions and take a little longer.

The answer starts with the direct answer, then the details, with markers like **[1]** after the sentences they support. Hover or select a marker to highlight its source. Each **source** shows the document title, page or section, a snippet, a score and a bar, and opens the full passage. Below the answer you see the provider, model, time taken and the knowledge version. **Thumbs up or down**, with an optional note, records feedback on that answer.

Answers come back in the language of the question.

From the command line:

```bash
ragkb ask "How many days of annual leave does a full-time employee get?" --collection handbook
ragkb ask "..." --collection handbook --provider extractive     # no AI
ragkb ask "..." --collection handbook --json                    # the full answer record
```

## 7. Reading the states and scores

Every answer has one of these states. They look different on purpose, so a "not covered" never reads like an answer.

| State | What happened | What to do |
|---|---|---|
| **Answered** | The provider wrote an answer from the passages. | Check the cited passages for anything important. |
| **Not covered** | The documents were searched, but none answers the question. Nothing is guessed. | Rephrase with the words the documents would use, raise **Passages**, or upload the document that has the answer. |
| **No documents** | The collection is empty, or nothing in it is indexed yet. | Upload documents, or wait until they are `ready`. |
| **Provider unavailable** | The chosen provider is not installed, not signed in, timed out or failed. The fix command is shown with a copy button. | Run the command (for example `claude auth login`), or pick another provider such as extractive mode. |
| **Rate limited** | Your plan's usage limit for that tool is reached. The vendor's message is shown. | Wait for the reset, or switch provider for now. |
| **Search unavailable** | The document store did not respond. | Check the health badge and the Settings tab; restarting the server usually helps. |

**Scores.** A source's score comes from rank fusion: a passage that ranks high in both the meaning search and the word search scores highest. Typical values are small (around 0.01 to 0.04); the score is a ranking signal, not a probability and not a measure of truth. The bar is relative to the strongest source in the same answer, so a full bar means "the best match here", not "a perfect match". Low scores across the board usually mean the collection does not cover the question well.

**Knowledge version.** A stamp like `v20261010.12` (date and a counter) that changes every time a document is added, replaced or deleted. Every answer carries the version it was answered from, so "which documents answered this" is always on record.

## 8. Managing documents and collections

In the documents table, each document has these actions:

- **Open passages**: the passages it was split into, in order, with their section and page.
- **Download the original file**: rag-kb keeps every original you upload.
- **Re-index**: parse and embed the document again from the kept original, for example after you change the chunking settings.
- **Delete**: removes the passages, the record and the original. Delete buttons need a second click within 4 seconds, so a single misclick does nothing.

**Delete collection** (top of the page) removes the collection with all its documents, using the same two-click confirmation.

Changing the embedding model (`RAGKB_EMBED_MODEL`) makes existing vectors incompatible. Start with a new data folder and upload again; if the new model has a different size, rag-kb refuses to mix them and says so.

## 9. Where the data lives, and backup

Everything rag-kb stores is in one **data folder**: `./data` under the folder you start it from, or the path in `RAGKB_DATA_DIR` (or `--data-dir` on any command). The Settings tab and the status bar show the exact path.

| Inside the data folder | Holds |
|---|---|
| `meta.sqlite` | Collections, documents and their status, upload jobs, stored answers, feedback, the audit log and the knowledge version. |
| `lancedb/` | The passages, their vectors and the full-text index. |
| `uploads/<collection>/<document>/` | The original files, as uploaded. |
| `logs/ragkb.log` | The server log. |

**Backup is the data folder.** Stop the server, copy the whole folder somewhere safe, start again. To restore, stop the server and put the copy back. To move rag-kb to another machine, copy the folder and point `RAGKB_DATA_DIR` at it.

The audit log records uploads, deletes and answers (who answered, which provider and model, how long it took, the outcome and the knowledge version), never the passages or the answer text.

Nothing leaves your machine except, for each question, the question and its selected passages, sent by Claude Code or Codex to your own subscription (the same as pasting them into that assistant). In extractive mode nothing leaves at all.

## 10. Settings

Settings come from environment variables starting with `RAGKB_`, or from a `.env` file in the folder you start rag-kb from. Copy `.env.example` to `.env` and edit; restart the server after a change. The ones you are most likely to touch:

| Setting | Default | What it does |
|---|---|---|
| `RAGKB_DATA_DIR` | `./data` | The data folder. |
| `RAGKB_PORT` | `8765` | The port of the web page, the API and `/mcp`. |
| `RAGKB_PROVIDER` | `auto` | `auto`, `claude_code`, `codex` or `extractive`. |
| `RAGKB_CLAUDE_MODEL` | `sonnet` | Default Claude model. |
| `RAGKB_CODEX_MODEL`, `RAGKB_CODEX_REASONING` | empty, `medium` | Codex model (empty means your Codex default) and reasoning effort. |
| `RAGKB_TOP_K` | `6` | Default passages per answer. |
| `RAGKB_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | English-first. For documents in several languages use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (also 384 dimensions; in a new data folder, or re-index every document). An unknown name fails at start-up with a list of supported models. |
| `RAGKB_RERANKER` | `none` | `local` adds a local re-ranking step. Measure it with the evaluation before keeping it. |
| `RAGKB_MAX_UPLOAD_MB` | `50` | Per-file upload limit. |
| `RAGKB_ANSWER_TIMEOUT_S` | `180` | How long to wait for Claude Code or Codex. |
| `RAGKB_MCP_ENABLED`, `RAGKB_MCP_PUBLIC` | `true`, `false` | The MCP endpoint for assistants, and whether it answers beyond this machine. |
| `RAGKB_API_PUBLIC` | `false` | Whether the API (and so the web page's data) answers beyond this machine. It has no sign-in: keep it `false` when a tunnel exposes `/mcp`. `GET /api/health` always answers. |

The full list, with comments, is in `.env.example`.

## 11. More

- [CONNECT-YOUR-ASSISTANT.md](CONNECT-YOUR-ASSISTANT.md): use the knowledge base from Claude Code, Codex, Claude Desktop or claude.ai.
- [PROVIDERS.md](PROVIDERS.md): how the subscriptions are used, what rag-kb can and cannot see, troubleshooting.
- [EVALS.md](EVALS.md): measure search quality and honesty on your own documents with `ragkb eval`.
