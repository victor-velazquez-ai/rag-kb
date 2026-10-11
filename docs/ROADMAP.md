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

- **Faster embedding.** Batch embedding now runs on several single-threaded model sessions at once (half the logical processors, at most eight), measured at 1.7x the speed of a single session. The embedding model itself is now the floor. Options left: a smaller or differently quantized embedding model (measured on the golden set before switching), less chunk overlap (fewer tokens to embed), or a GPU build of the ONNX runtime.
- Order the extractive digest by sentence weight, not by search rank, so the strongest sentence leads.
- Tighten extractive declines on lexical near-misses (a question that shares real words with the corpus but asks something it does not cover).
- OCR for scanned PDFs (a local engine, optional dependency).
- Streaming answers in the UI (the contract allows it; the composer already holds back the sentinel).
- Multilingual default: measure `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` against the English default on a mixed-language golden set.
- **Per-collection embedding models.** Let each collection pick its embedding model (English-first or multilingual), store the model with the collection's vectors, and embed questions with the model of the collection asked. Switching a collection's model then re-embeds only that collection, as a tracked job with progress and cancel, while the old vectors keep answering until the new ones are ready. Today the model is one setting for the whole data folder, and changing it means re-indexing every document by hand.
- Re-index a whole collection in one step (the Documents tab and `ragkb reindex --collection`), for chunking or model changes.
- A golden-set builder: draft questions from a corpus with the person's own provider, for them to review.

## 3. Product

- Per-document metadata editing (title, tags, source URL) and tag filters in search.
- Scheduled re-ingestion from a watched folder or a shared drive.
- Export and import of a collection (originals plus metadata) for moving between machines.
- Answer history per collection with search.
- From feedback to evaluation: turn a "Not helpful" vote into a golden-set case in one step, with the answer's closest passages as the starting point.
- Verify the Claude Desktop setup end to end in the real application, and the claude.ai custom connector through a tunnel.

## 4. Objective 2: the web product

In the order of `docs/MASTER-PLAN.md` section 4: sign-in and organizations with server-side roles; Postgres with pgvector behind the same store interface and object storage for originals; OAuth for the MCP server (client registration, redirect allow-list, hashed tokens, session cap, revoke by user, a sign-in page that names the connecting app); the answer brief that lets the person's own claude.ai or ChatGPT write the final answer; Docker, health, metrics, backups, rate limits, a deploy runbook; then streaming, OCR, scheduled ingestion and a local-bridge mode where the web UI chats through the person's own CLI on their machine.

## Done

- QA round (10 October 2026), ingestion and answers: every passage fits the embedding model's window (counted with the model's own tokenizer, with a hard ceiling on passage length), so nothing is embedded truncated; hidden spreadsheet sheets are left out; a password-protected PDF, Word or Excel file is refused up front with `encrypted_file` and leaves no row; two ingestion workers, so small files are not stuck behind large ones; batch embedding on parallel model sessions (1.7x; see "Faster embedding" above for what is left); the local reranker on by default, with calibrated relevance bands that decline a question whose passages are all weak before any model call; a pool of 30 candidates, rarity-weighted rank fusion and a boost for exact identifiers and quoted phrases (`FH-MNT-01`, `"core hours"`); answers get every passage that fits the context budget plus the neighbours of the best ones, so a table or procedure split across passages comes through whole; a model the provider does not know is refused before any call with a plain message; deletes run apart from the ingestion workers and answer 202 while they finish in the background; `ragkb doctor` downloads and checks the reranker as well as the embedding model; the evaluation runs the same retrieval as the app.
- QA round (10 October 2026), web app: documents keep their path as identity (titles may repeat); folder and `.zip` upload; per-file job progress with stage, passage counts and time left; Cancel on queued and running jobs; Retry and Dismiss on failed documents; a queue summary; searchable documents as the headline count everywhere; page spans on passages and citations; raw similarity and reranker scores next to the rank score, with an absolute bar; the closest passages under a "not covered" answer; feedback review for curators on the Settings tab; addresses for every tab, collection and answer with Back and Forward; a curl, PowerShell or cmd version of each request; a documents table that fits its container at any width.
- v0.1.0 (10 October 2026): objective 1. Hybrid retrieval, structure-aware chunking, three providers (Claude Code, Codex, extractive), the honest answer contract, the web UI, the MCP server, the evaluation harness, user docs, assistant setup instructions.
