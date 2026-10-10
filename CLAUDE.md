# Working on rag-kb with an AI coding assistant

This file is for the assistant that opens this repository (Claude Code, Codex, Cursor, or any other agent). It tells you what the project is, how to get it running for the person you are helping, and the rules of the codebase. `AGENTS.md` points here, so every tool reads the same instructions.

## What this is

rag-kb is a local knowledge base: people upload documents, the app vectorizes them on this machine, and questions get a readable, cited answer written by the person's **own Claude or ChatGPT subscription** through the vendors' command-line tools. No API keys are involved, and no cloud AI service is called by this app. Everything about the design is in `docs/MASTER-PLAN.md` and `docs/ARCHITECTURE.md`.

## Your first job after a clone: make it run

Do these in order, and tell the person what you did and what is left for them. Stop and ask only when a step needs their account (signing in) or their choice.

1. **Check Python.** `python --version` must be 3.11 or newer. On Windows the launcher may be `py -3.12`. If Python is missing, point them at python.org and stop; do not install a system Python for them without asking.
2. **Create the environment and install.**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate       macOS/Linux: source .venv/bin/activate
   pip install -e ".[dev]"
   ```
   On Windows, run the project's own interpreter explicitly when the shell does not activate: `.venv\Scripts\python.exe -m ragkb.cli ...`.
3. **Run the health check.** `ragkb doctor` (or `python -m ragkb.cli doctor`). It reports Python, the embedding model (downloaded on the first run, about 130 MB, from the Hugging Face hub; it needs internet once), the data folder, and each answer provider: installed or not, signed in or not, and the exact command that fixes it.
4. **Set up the subscription provider the person has.** Only one is needed; both can coexist.
   - **Claude Pro or Max:** Claude Code must be installed and signed in. Install: `npm install -g @anthropic-ai/claude-code` (needs Node.js 18 or newer), or the native installer from the Claude Code documentation. Sign in: `claude auth login` (opens the browser). The VS Code extension's bundled `claude` binary is detected automatically on Windows, so people who already use that extension need nothing else.
   - **ChatGPT Plus or Pro:** Codex must be installed and signed in. Install: `npm install -g @openai/codex`. Sign in: `codex login`. The VS Code ChatGPT extension's bundled `codex` is detected automatically on Windows.
   - **Neither:** the app still works in extractive mode (a cited digest of the best sentences, no AI). Say so plainly instead of blocking.
   Signing in is the one thing you cannot do for them: it needs their browser and their account. Give them the command, wait, then rerun `ragkb doctor`.
5. **Run the tests once**: `python -m pytest tests -q`. They are offline and take under a minute. A failure here is an environment problem worth fixing before anything else (most often a missing dependency or an old Python).
6. **Start it**: `ragkb serve --open`. The app opens at http://127.0.0.1:8765. Confirm `/api/health` answers `"status": "ok"`.
7. **Prove it end to end** with the sample corpus, then show the person how to add their own documents:
   ```bash
   ragkb ingest samples/corpus --collection sample
   ragkb ask "How many days of annual leave does a full-time employee get?" --collection sample
   ragkb ask "Can I bring my dog to work?" --collection sample     # must come back as not covered
   ```
8. **Connect their assistant to the knowledge base, if they want it** (`docs/CONNECT-YOUR-ASSISTANT.md`): `claude mcp add --transport http rag-kb http://127.0.0.1:8765/mcp` for Claude Code, `codex mcp add rag-kb --url http://127.0.0.1:8765/mcp` for Codex.

## Things that go wrong, and the fix

| Symptom | Cause and fix |
|---|---|
| `ragkb` is not a command | The environment is not activated; use `python -m ragkb.cli` or activate `.venv`. |
| Claude Code answers fail with "not signed in" although `claude auth status` says signed in | Something adds `--bare`, or `ANTHROPIC_API_KEY` or a Bedrock variable is set. rag-kb removes those for its own calls; check wrappers and aliases. |
| Codex answers fail with a quota or rate message | The ChatGPT plan's Codex quota is small. Wait, or switch the provider to Claude Code or extractive in the Ask tab. |
| The embedding model will not download | No internet, or a proxy. The hub cache is `fastembed_cache` in the temp folder; a corporate proxy needs `HTTPS_PROXY`. |
| Port 8765 is busy | `ragkb serve --port 8800`. |
| 403 on `/api` or `/mcp` | The request did not come from localhost. That is deliberate; see `RAGKB_API_PUBLIC` and `RAGKB_MCP_PUBLIC` in `.env.example`, and read the warning there before enabling them. |
| A PDF is rejected as empty | It is a scanned image with no text layer. OCR is not built in yet. |
| On Windows, an empty command-line argument vanishes | PowerShell 5.1 drops empty-string arguments. rag-kb passes arguments as lists from Python, so this only matters if you call the CLIs by hand. |

## Rules of this codebase

- **Read `docs/ARCHITECTURE.md` before changing an interface.** The contracts there are what the parts are built against; change the document first, then the code, then the tests.
- **Tests must stay green and offline by default**: `python -m pytest tests -q` and `python -m ruff check .` before you consider a change done. Live tests (real models, real CLIs) are opt-in.
- **Every setting is documented.** A test checks that every `Settings` field in `ragkb/config.py` appears in `.env.example` with the code's default, and the reverse. Add both or neither.
- **Honesty contract.** A declined answer has `status: no_answer` and `answer: null`. Never turn a decline into text that could be mistaken for an answer, and never paste raw passages into a user-facing reply outside the composer.
- **Tenancy.** Every store read and every destructive statement is filtered by collection. The tests treat isolation as a hard gate.
- **No secrets, no keys.** The app holds no credentials; the vendors' tools hold their own sign-ins. Do not add API-key paths as a requirement.
- **No external resources in the UI** (no CDN scripts or styles), and no build step.
- **Plain language in messages and docs.** Error messages are one sentence, with no file paths or stack traces. Prefer commas and colons to long dashes.
- **Data is local.** `data/` is ignored by git and holds the person's documents and index. Never commit it, never delete it without asking; backing up means copying that folder with the server stopped.
- **Commits.** Small, by area, with a one-line subject that says what changed and why. No tool attribution lines.

## Where things live

| Folder | What |
|---|---|
| `ragkb/ingest` | parsers, chunker, pipeline (upload to index) |
| `ragkb/store` | LanceDB chunks and vectors, SQLite metadata |
| `ragkb/retrieval` | embedders, reranker, hybrid retriever |
| `ragkb/answer` | prompts, composition, citations |
| `ragkb/providers` | Claude Code, Codex and extractive adapters, registry |
| `ragkb/server` | FastAPI app, routes, dependency container |
| `ragkb/mcp` | the MCP server at `/mcp` |
| `ragkb/evals` | golden set, scorers, runner |
| `ragkb/ui` | the single-page web app |
| `samples/` | a fictional sample corpus and its golden set |
| `docs/` | plan, architecture, user guide, providers, connect, evals, design lessons |
