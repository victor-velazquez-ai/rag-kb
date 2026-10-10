# rag-kb

An enterprise knowledge base with retrieval-augmented answers. Upload a document corpus, let it be vectorized locally, and ask questions. You get a readable, organized, cited answer, written by **your own Claude or ChatGPT subscription**. No AI API keys, no cloud AI bill.

- **Your subscription, your answers.** The app drives the official command-line tools that come with Claude Pro/Max (Claude Code) and ChatGPT Plus/Pro (Codex). Sign in to them once, as you already do, and every answer runs on your plan. Or connect Claude Desktop, claude.ai, Claude Code or Codex to the knowledge base through MCP and ask there.
- **Everything else is local.** Parsing, embeddings, search and the vector store run on your machine with open models. Your documents stay with you; only the passages chosen for a question go to your own assistant.
- **Honest.** When the documents do not answer, it says so as a distinct "not covered" state, instead of guessing. An evaluation harness measures it.
- **Works without a subscription too.** Extractive mode answers with a cited digest of the best sentences and no AI.

## Quickstart

You need Python 3.11 or newer.

```bash
git clone https://github.com/victor-velazquez-ai/rag-kb.git
cd rag-kb
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .
ragkb doctor          # checks Claude Code / Codex sign-in, the embedding model (downloaded on first run), the data folder
ragkb serve --open    # opens http://127.0.0.1:8765
```

Then create a collection in the sidebar, drop your documents in, and ask on the Ask tab.

For AI-written answers, sign in once to the tool that matches your plan: `claude auth login` (Claude Pro or Max) or `codex login` (ChatGPT Plus or Pro). `ragkb doctor` shows what is missing and the command that fixes it. See [docs/PROVIDERS.md](docs/PROVIDERS.md).

### Try it on the sample corpus

`samples/corpus` holds ten short documents of a fictional company (a handbook, policies, a product FAQ, a procedure, a holiday calendar, a price list) in Markdown, text, HTML, CSV, Word, PDF and Excel.

```bash
ragkb ingest samples/corpus --collection sample
ragkb ask "How many days of annual leave does a full-time employee get?" --collection sample
ragkb ask "Can I bring my dog to work?" --collection sample           # not covered: says so
ragkb ask "How long do we keep CCTV footage?" --collection sample --provider extractive
```

## Use it from your assistant (MCP)

While `ragkb serve` runs, the knowledge base is also an MCP server at `http://127.0.0.1:8765/mcp` with four read-only tools: `list_collections`, `list_documents`, `search_knowledge_base` and `ask_knowledge_base`. Your assistant searches through them and writes the answer on your subscription.

```bash
claude mcp add --transport http rag-kb http://127.0.0.1:8765/mcp     # Claude Code
codex mcp add rag-kb --url http://127.0.0.1:8765/mcp                  # Codex
```

Claude Desktop (through the `mcp-remote` bridge) and claude.ai (through a tunnel) are covered step by step in [docs/CONNECT-YOUR-ASSISTANT.md](docs/CONNECT-YOUR-ASSISTANT.md). The endpoint answers only on localhost unless you set `RAGKB_MCP_PUBLIC=true`; the API stays local-only either way unless you set `RAGKB_API_PUBLIC=true`.

## Measure it

```bash
ragkb eval            # offline: deterministic, no model, runs in seconds
ragkb eval --live     # the real embedding model and your answer provider (one call per question)
```

The evaluation indexes the golden set's corpus into a throwaway folder, asks every question and reports recall@k, MRR, citation rate and the false-answer rate on questions that must be declined, with a PASS, FAIL or SKIPPED verdict. Write a golden set for your own documents with [docs/EVALS.md](docs/EVALS.md).

## Where your data is

Everything lives in one data folder, `./data` by default (`RAGKB_DATA_DIR` or `--data-dir` to change it): the metadata database, the vector index, your original files and the log. **Backing up means copying that folder** while the server is stopped. Settings come from `RAGKB_*` environment variables or a `.env` file; see `.env.example`.

## Commands

| Command | Does |
|---|---|
| `ragkb serve [--open] [--port N]` | Runs the web app, the API (`/api`) and the MCP endpoint (`/mcp`). |
| `ragkb ingest PATH... --collection NAME` | Indexes files and folders into a collection (created when missing). |
| `ragkb ask "QUESTION" [--collection NAME] [--provider auto\|claude_code\|codex\|extractive] [--model M] [--json]` | Prints a cited answer. |
| `ragkb doctor [--json]` | Checks the installation, providers and stores, with fix commands. |
| `ragkb eval [--golden FILE] [--live] [--json]` | Runs the evaluation. |

Every command takes `--data-dir`.

## Documents

- [docs/USER-GUIDE.md](docs/USER-GUIDE.md): install, sign in, collections, uploads, asking, the answer states and scores, re-index and delete, data and backup.
- [docs/PROVIDERS.md](docs/PROVIDERS.md): how each subscription is used, the exact commands rag-kb runs, what it can and cannot see, troubleshooting.
- [docs/CONNECT-YOUR-ASSISTANT.md](docs/CONNECT-YOUR-ASSISTANT.md): Claude Code, Codex, Claude Desktop and claude.ai setup, the MCP tools.
- [docs/EVALS.md](docs/EVALS.md): the golden-set format, the scorers, reading the report.
- [docs/MASTER-PLAN.md](docs/MASTER-PLAN.md): vision, how subscriptions power the answers, objectives, stack, phases.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the contracts every part is built against.
- [docs/design-heritage.md](docs/design-heritage.md): lessons from earlier knowledge-base work that shaped this design.

## Status

Objective 1 (run it locally, one user, your own subscription) is built. Objective 2 (one web deployment for many users, with sign-in and roles, where each user's claude.ai or ChatGPT connects through MCP) is designed in the master plan and not built yet. Run the tests with `pip install -e ".[dev]"` and `pytest`.
