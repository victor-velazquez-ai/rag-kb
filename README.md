# rag-kb

An enterprise knowledge base with retrieval-augmented answers. Upload a document corpus, let it be vectorized locally, and ask questions. You get a readable, organized, cited answer, written by **your own Claude or ChatGPT subscription**. No AI API keys, no cloud AI bill.

- **Your subscription, your answers.** The app drives the official command-line tools that come with Claude Pro/Max (Claude Code) and ChatGPT Plus/Pro (Codex). Sign in to them once, as you already do, and every answer runs on your plan. On the web (objective 2), your claude.ai or ChatGPT connects to the knowledge base directly through MCP.
- **Everything else is local.** Embeddings, search and the vector store run on your machine with open models. Your documents stay with you.
- **Honest.** When the documents do not answer, it says so, instead of guessing.

Status: objective 1 (run it locally) is being built. See `docs/MASTER-PLAN.md`.

## Quickstart

```bash
git clone https://github.com/victor-velazquez-ai/rag-kb.git
cd rag-kb
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ragkb doctor          # checks Claude Code / Codex sign-in, embedding model, data folder
ragkb serve --open    # opens http://127.0.0.1:8765
```

Then create a collection, drop your documents in, and ask.

## Documents

- `docs/MASTER-PLAN.md`: vision, how subscriptions power the answers, objectives, stack, phases.
- `docs/ARCHITECTURE.md`: the contracts every part is built against.
- `docs/design-heritage.md`: lessons from earlier knowledge-base work that shaped this design.
