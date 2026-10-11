# rag-kb user guide

rag-kb turns a folder of documents into a knowledge base you can question. You upload the documents, rag-kb splits them into passages and indexes them on your machine, and every question gets a direct, readable answer with numbered citations back to the passages it came from. The answer is written by your own Claude or ChatGPT subscription; when the documents do not answer the question, rag-kb says so instead of guessing.

## 1. What you need

- **Python 3.11 or newer** (`python --version`).
- About **1 GB of free disk**: the embedding model is about 130 MB and the reranker about 90 MB (both downloaded once, on the first run), the rest is your documents and their index.
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

`doctor` checks Python, the data folder, the stores, the embedding model and the reranker, the answer providers and the MCP server. The first run downloads the two local models from the Hugging Face hub (the embedding model about 130 MB, the reranker about 90 MB; it needs internet once), so the first question does not wait on them. Every line should read `[  ok]`. A provider that is not ready shows the exact command that fixes it.

```bash
ragkb serve --open
```

This starts rag-kb at **http://127.0.0.1:8765** and opens your browser. Leave the terminal open; Ctrl+C stops the server. `--port 9000` picks another port.

The page follows your system's light or dark setting; the **Theme** switch at the foot of the sidebar picks **System**, **Light** or **Dark**, and this browser remembers the choice. On a reload the health and provider badges show their last known state at once, then refresh.

Start rag-kb from the same folder each time (or set `RAGKB_DATA_DIR` to an absolute path, see section 9): the data folder is `./data` relative to where you start it.

## 5. Create a collection and upload

A **collection** holds one corpus: a team's handbook, a product's manuals, a client's contracts. Questions are answered from one collection at a time, and collections never see each other's documents.

1. In the left sidebar, under **New collection**, type a name (and optionally a description) and choose **Create collection**.
2. On the **Documents** tab, drop files on the drop zone or choose **choose files**. You can add many at once.
   - **A whole folder:** choose **Add folder**, or drop the folder on the drop zone. Every supported file in it and in its subfolders is uploaded; hidden files and other file types (pictures, for example) are left out, and one line under **Uploads** says how many.
   - **A `.zip`:** drop it or choose it like any file. rag-kb unpacks it and adds one document per file inside.
   - Types: PDF, Word (`.docx`), Markdown, text, HTML, CSV and TSV, Excel (`.xlsx`), and `.zip` archives of them. Older `.doc` and `.xls` files must be saved as `.docx` or `.xlsx` first.
   - Size: up to 50 MB per file (`RAGKB_MAX_UPLOAD_MB`).
   - **Title** (optional, one file only): the name shown in citations. Otherwise the document's own title or the file name is used.
3. Each file gets a row under **Uploads** with a progress bar, its status and what is happening now, for example "Embedding 120 of 300 passages, about 40 s left". **Cancel** stops a file that is waiting, uploading or being indexed. When it is done, the document appears in the documents table:

| Status | Meaning |
|---|---|
| queued | Waiting for a free worker. |
| processing | A worker is reading it, splitting it into passages and embedding them. |
| ready | Indexed and searchable. |
| failed | Not indexed; the reason is shown (for example "This PDF has no extractable text": scanned PDFs need OCR, which this version does not do). |
| cancelled | Stopped with **Cancel** before it was indexed. |

An upload row can also say:

| Upload row | Meaning |
|---|---|
| **Already indexed** | The same content is already in this collection, so it was kept once. Nothing is duplicated. |
| **Replaced "Old title"** | A document with the same file name (or the same path in a folder or zip) existed; the new content replaced it. |
| **Rejected:** and a reason | The file was not accepted, for example an unsupported type, an empty file or one over the size limit. |
| **Unpacked** | A `.zip` was opened; the rows under it are the files it held. |

**How rag-kb tells documents apart.** A document is known by its file name, or by its path when it came from a folder or a zip (`handbooks/north/policy.pdf` and `handbooks/south/policy.pdf` are two documents). Uploading the same name or path again with new content replaces the earlier version, and the old version stays searchable until the new one is fully indexed, so a failed or cancelled upload never loses the document you had. Titles are only labels: two documents may have the same title.

Above the table, one line sums up the uploads of this collection, for example "Uploads: 2 processing, 42 ready, 3 failed, 1 replaced, 2 duplicates".

From the command line, the same thing:

```bash
ragkb ingest path/to/folder another-file.pdf --collection handbook
```

Folders are read recursively; the collection is created when it does not exist.

## 6. Ask

Open the **Ask** tab, type a question and press Enter (Shift+Enter adds a line; `/` anywhere jumps back to the question box).

- **Provider**: Auto (Claude Code if signed in, then Codex, then extractive), or pick one for this question.
- **Model**: for example `sonnet`, `opus` or `haiku` for Claude Code; Codex offers the models your install knows. Extractive mode uses no model, so the menu is disabled and says "no model".
- **Passages**: how many passages are given to the provider (default 6). More passages help broad questions and take a little longer.

The answer starts with the direct answer, then the details, with markers like **[1]** after the sentences they support. Hover or select a marker to highlight its source. Each **source** shows the document title, the page (or "pages 3 to 4" when the passage runs over a page break) and section, a snippet, its scores, and opens the full passage. Below the answer you see the provider, model, time taken and the knowledge version. **Helpful** or **Not helpful**, with an optional note, records feedback on that answer; curators read it on the Settings tab (section 8).

**Raw response and request**, under each answer, shows the full answer record and the same request as a terminal command. Pick **bash** (macOS, Linux, Git Bash), **PowerShell** or **cmd** (Windows); the browser remembers the choice. The question is written with JSON escapes for anything but letters, digits and plain punctuation, so the command works as pasted in each of them.

Every view has its own address: `#/c/<collection>/<tab>` for a tab of a collection and `#/answer/<id>` for an answer. The browser's Back and Forward move between tabs, collections and answers, a reload opens the same view, and an answer's address can be bookmarked.

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
| **Not covered** | The documents were searched, but none answers the question. Nothing is guessed. Underneath, **Closest passages** shows the nearest ones found, so you can judge for yourself. | Read the closest passages. Rephrase with the words the documents would use, raise **Passages**, or upload the document that has the answer. |
| **No documents** | The collection is empty, or nothing in it is indexed yet. | Upload documents, or wait until they are `ready`. |
| **Provider unavailable** | The chosen provider is not installed, not signed in, timed out or failed. The fix command is shown with a copy button. | Run the command (for example `claude auth login`), or pick another provider such as extractive mode. |
| **Rate limited** | Your plan's usage limit for that tool is reached. The vendor's message is shown. | Wait for the reset, or switch provider for now. |
| **Search unavailable** | The document store did not respond. | Check the health badge and the Settings tab; restarting the server usually helps. |

**Scores.** The rank score orders the sources; the similarity and reranker scores say how relevant each one is.

| Score | What it is | How to read it |
|---|---|---|
| **similarity** (and the bar) | The cosine similarity between the meaning of the question and of the passage, from 0 to 1. | The same scale in every answer, so a full bar is a close match anywhere. What counts as high depends on the embedding model; compare sources with each other and learn what a good match looks like on your documents. A passage found only by the word search shows "no similarity score". |
| **reranker** | Shown when the local reranker is on (`RAGKB_RERANKER=local`): a second model reads the question and the passage together. | Higher means more relevant. |
| **rank** | The rank fusion score: a passage that ranks high in both the meaning search and the word search comes first. | Only the order. Typical values are small (around 0.01 to 0.04); it is not a probability and not a measure of truth. |

Low similarity across all sources usually means the collection does not cover the question well.

**Knowledge version.** A stamp like `v20261010.12` (date and a counter) that changes every time a document is added, replaced or deleted. Every answer carries the version it was answered from, so "which documents answered this" is always on record.

## 8. Managing documents and collections

The number next to a collection in the sidebar, and the first number under its name, is the count of **searchable** documents (status ready). Documents still processing, failed or cancelled are listed after it as a note, for example "12 searchable documents (1 processing, 2 failed)".

In the documents table, under each title, the file name (or its path) downloads the original file: rag-kb keeps every original you upload. The type, size and, on narrow screens, the pages and dates are on the same line. Each document has these actions:

- **Passages**: the passages it was split into, in order, with their section and pages.
- **Re-index**: parse and embed the document again from the kept original, for example after you change the chunking settings or the embedding model.
- **Delete**: removes the passages, the record and the original.
- On a **failed** or **cancelled** document: **Retry** indexes it again from the kept original, and **Dismiss** removes it.
- On a **queued** or **processing** document: **Cancel** stops the job.

Delete and Dismiss need a second click within 4 seconds, so a single misclick does nothing. While a delete, re-index or retry runs, the row shows a spinner, and a note when it takes more than two seconds.

**Delete collection** (top of the page) removes the collection with all its documents, using the same two-click confirmation.

**Feedback review.** The **Settings** tab has a **Feedback** panel with the votes left on this collection's answers, newest first: the question, Helpful or Not helpful, the note and when. **Open answer** shows the answer as it was given, with its sources. "Not helpful" votes are the best place to start improving a collection, and good candidates for an evaluation case ([EVALS.md](EVALS.md)).

### Documents in other languages

The default embedding model, `BAAI/bge-small-en-v1.5`, is **English-first**. Answers always come back in the language of the question, but search for other languages is weaker: questions and documents in other languages find fewer of the right passages, and a question asked in one language rarely finds a passage written in another.

Switch to the multilingual model, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, when a real share of your documents or of your users' questions is not in English, or when people ask in one language about documents written in another. For a corpus that is all English, keep the default: it is the better English model.

Switching means **every document must be embedded again**, because vectors from two models cannot be compared:

1. Stop the server and set `RAGKB_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in `.env`.
2. Either start with a new, empty data folder (`RAGKB_DATA_DIR`) and upload your documents again, which is simplest, or keep the data folder and choose **Re-index** on every document of every collection.

Both models have 384 dimensions, so rag-kb cannot tell old vectors from new ones and will not stop you: until every document is re-indexed, search quality is poor. Uploading the same files again does not re-embed them (the same content is kept once), so use Re-index or a new data folder. A model with a different size is refused at start-up with a message that says so. Choosing the embedding model per collection is on the [roadmap](ROADMAP.md#2-quality).

## 9. Where the data lives, and backup

Everything rag-kb stores is in one **data folder**: `./data` under the folder you start it from, or the path in `RAGKB_DATA_DIR` (or `--data-dir` on any command). The Settings tab and the status bar show the exact path.

| Inside the data folder | Holds |
|---|---|
| `meta.sqlite` | Collections, documents and their status, upload jobs and their progress, stored answers, feedback, the audit log and the knowledge version. |
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
| `RAGKB_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | English-first. For documents or questions in other languages use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; every document must then be embedded again (see "Documents in other languages" in section 8). An unknown name fails at start-up with a list of supported models. |
| `RAGKB_RERANKER` | `local` | `local` re-orders the passages with a small local model that reads the question and each passage together (about 90 MB, downloaded once); when it scores every passage weak, the question is declined without calling Claude Code or Codex. `none` keeps the plain search order. |
| `RAGKB_MAX_UPLOAD_MB` | `50` | Per-file upload limit. |
| `RAGKB_ANSWER_TIMEOUT_S` | `180` | How long to wait for Claude Code or Codex. |
| `RAGKB_MCP_ENABLED`, `RAGKB_MCP_PUBLIC` | `true`, `false` | The MCP endpoint for assistants, and whether it answers beyond this machine. |
| `RAGKB_API_PUBLIC` | `false` | Whether the API (and so the web page's data) answers beyond this machine. It has no sign-in: keep it `false` when a tunnel exposes `/mcp`. `GET /api/health` always answers. |

The full list, with comments, is in `.env.example`.

## 11. More

- [CONNECT-YOUR-ASSISTANT.md](CONNECT-YOUR-ASSISTANT.md): use the knowledge base from Claude Code, Codex, Claude Desktop or claude.ai.
- [PROVIDERS.md](PROVIDERS.md): how the subscriptions are used, what rag-kb can and cannot see, troubleshooting.
- [EVALS.md](EVALS.md): measure search quality and honesty on your own documents with `ragkb eval`.
