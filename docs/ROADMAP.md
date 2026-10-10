# Roadmap and future tasks

What is not built yet, in rough priority order. Objective 1 (run it locally) shipped as v0.1.0; objective 2 (one web deployment for many users) is designed in `docs/MASTER-PLAN.md` section 4.

## 1. Bring any subscription

The answer providers are adapters behind one small interface (`ragkb/providers/base.py`: `status()` and `generate()`), and the registry picks the first signed-in one. Adding a subscription means one adapter file, its tests against recorded CLI output, and a row in `docs/PROVIDERS.md`. Candidates, each already shipping an official command-line tool that signs in with the consumer account:

| Subscription | Tool to wrap | Notes |
|---|---|---|
| Google AI Pro / Ultra (Gemini) | Gemini CLI (`gemini`), sign in with a Google account | Has a non-interactive prompt mode; confirm a JSON or plain output mode and how quota errors surface. |
| GitHub Copilot | Copilot CLI (`copilot` or `gh copilot`) | Signs in with the GitHub account; check that a plain completion mode exists without repository context. |
| Mistral | Mistral's CLI or Vibe tooling | Verify a subscription-backed headless mode exists before promising it. |
| Any OpenAI-compatible endpoint | A small HTTP adapter (`base_url`, optional key) | For teams that run their own gateway or a local model server (Ollama, LM Studio, vLLM). Optional, never required. |
| Local open models | Ollama (`ollama run`) | A true no-key, offline mode with real generation; quality depends on the model and the machine. |

Design rules for every adapter:
- The prompt goes on stdin or in a temp file, never on the command line (Windows length limits).
- Strip the vendor's API-key variables from the child process so the call runs on the subscription, not on a key someone has set.
- Map the vendor's errors to the five `ProviderError` kinds; map quota messages to `rate_limited` with the vendor's text.
- `status()` must answer "installed, signed in, version, fix command" without spending a model call.
- Record one real success and one real not-signed-in output as test fixtures, scrubbed of ids and emails.
- Prove the system prompt is honoured: the sentinel decline (`NO_ANSWER`) must come back on a question the passages do not answer.

Also for the provider layer: let the person set a default provider and model per collection, show provider quota or usage hints in Settings when a tool exposes them, and add a "compare providers" button on the Ask tab that runs the same question through two providers side by side.

## 2. Quality

- Measure the local reranker (`RAGKB_RERANKER=local`) on the golden set and turn it on by default if it helps.
- Order the extractive digest by sentence weight, not by search rank, so the strongest sentence leads.
- Tighten extractive declines on lexical near-misses (a question that shares real words with the corpus but asks something it does not cover).
- OCR for scanned PDFs (a local engine, optional dependency).
- Streaming answers in the UI (the contract allows it; the composer already holds back the sentinel).
- Multilingual default: measure `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` against the English default on a mixed-language golden set.
- A golden-set builder: draft questions from a corpus with the person's own provider, for them to review.

## 3. Product

- Per-document metadata editing (title, tags, source URL) and tag filters in search.
- Scheduled re-ingestion from a watched folder or a shared drive.
- Export and import of a collection (originals plus metadata) for moving between machines.
- Answer history per collection with search, and feedback review for curators.
- Verify the Claude Desktop setup end to end in the real application, and the claude.ai custom connector through a tunnel.

## 4. Objective 2: the web product

In the order of `docs/MASTER-PLAN.md` section 4: sign-in and organizations with server-side roles; Postgres with pgvector behind the same store interface and object storage for originals; OAuth for the MCP server (client registration, redirect allow-list, hashed tokens, session cap, revoke by user, a sign-in page that names the connecting app); the answer brief that lets the person's own claude.ai or ChatGPT write the final answer; Docker, health, metrics, backups, rate limits, a deploy runbook; then streaming, OCR, scheduled ingestion and a local-bridge mode where the web UI chats through the person's own CLI on their machine.

## Done

- v0.1.0 (10 October 2026): objective 1. Hybrid retrieval, structure-aware chunking, three providers (Claude Code, Codex, extractive), the honest answer contract, the web UI, the MCP server, the evaluation harness, user docs, assistant setup instructions.
