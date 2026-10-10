# Design heritage: what earlier knowledge-base work taught us

rag-kb is a fresh implementation, but it stands on lessons from building and operating a production knowledge base for an AI assistant over several months. These are the ones that shaped this design.

## Keep

1. **Honesty is a contract, not a prompt.** A model told to "say you don't know" sometimes writes text that reads like an answer. Use an exact sentinel the code can detect, turn it into a status, and render that status distinctly. Every response carries every field, always, so clients never guess.
2. **Hybrid retrieval with rank fusion.** Vector search finds meaning; full-text search finds names, codes and exact phrases. Fuse the two ranked lists (reciprocal rank fusion) and drop near-duplicates (same text, or a neighbouring chunk of the same document that repeats its text) before composing. A neighbour with its own text is usually the next section, not a copy, so it stays.
3. **Composition is the relevance filter.** A cosine floor could not separate good from bad passages: the score distributions overlapped. Keep the floor at zero by default and let the composition step decide what answers the question.
4. **Budget once, cite only what was kept.** Select the passages that fit the context budget once; number only those; a dropped passage can never be cited.
5. **Dense, stable citations.** Renumber markers to 1..K in order of first use and deduplicate by source. Gaps and repeats broke clients before.
6. **Knowledge version on everything.** A `v{YYYYMMDD}.{n}` stamp on every chunk and every answer makes "which corpus answered this" a fact, not a guess.
7. **Write everything, then switch.** Embed all chunks first, write in one transaction, and only then remove the stale tail of a replaced document. Never delete first.
8. **Tenancy on every statement.** Every read and every destructive statement carries the collection (or tenant) filter, and the tests treat isolation as a hard gate.
9. **Degrade, never block.** Search unavailable, provider unavailable, reranker failing: each has a defined fallback and a status the UI understands.
10. **Measured gates that cannot lie.** An evaluation that measured nothing must report "skipped", never "pass". Score recall over positives only, false answers over negatives only.
11. **Admin surfaces default off** outside local use, and roles are resolved on the server, never read from a client header.
12. **UI details that mattered:** a two-click delete that disarms itself, inline passage drill-down, a visible store-health banner, suggested questions that include an out-of-scope probe, a "how to read the scores" note, and a "connect your assistant" panel with copyable setup.

## Avoid

1. **Structure-blind chunking.** Fixed word windows split definitions mid-sentence and lost page and section metadata. Pack structural blocks instead, and keep the heading path and page on every chunk.
2. **Building without wiring.** A circuit breaker, rate limits, an approval gate, a feedback consumer and version history all existed with no callers. Every component here has a caller and a test.
3. **Raw passages in user replies.** Users mistook pasted passages for answers. Only the composer renders passage text, and only inside a prompt.
4. **Glossary-driven query expansion.** Expanding questions from a glossary asserted false things about names (surnames read as trades, initials as acronyms). Curated terms, applied narrowly, are safer.
5. **Shared connections across threads, and a retriever built per request.** Pool connections, build the retriever once.
6. **Deleting the original upload.** Keep it; it is the backup and the input for re-indexing.
7. **Per-process limits presented as global.** Say what a limit really covers.
8. **Same-model judges as a quality metric.** They are a smoke signal. Measure retrieval with a golden set and read a sample of answers.
