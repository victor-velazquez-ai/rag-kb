# Answer providers: how your subscription writes the answers

rag-kb does the searching itself, on your machine. Only the last step, turning the best passages into a readable answer, uses an AI model, and that model is reached through **your own subscription** by running the vendor's official command-line tool. rag-kb holds no API key and never sees your password or login token.

There are three providers:

| Provider | Setting value | Needs | Writes the answer with |
|---|---|---|---|
| Claude Code | `claude_code` | Claude Pro or Max (or a Team or Enterprise seat with Claude Code), the `claude` tool signed in | Claude (default model `sonnet`) |
| Codex | `codex` | ChatGPT Plus or Pro (or a plan that includes Codex), the `codex` tool signed in | Your Codex default model |
| Extractive mode | `extractive` | Nothing | No model: a digest of the most relevant sentences |

`RAGKB_PROVIDER=auto` (the default) uses Claude Code when it is installed and signed in, otherwise Codex when it is, otherwise extractive mode. You can also choose per question in the Ask tab or with `ragkb ask --provider codex "..."`. `ragkb doctor` and the Settings tab show which providers are ready and the exact command to fix each one that is not.

## Claude Code (Claude subscription)

**Install** one of these ways:

```powershell
# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```
```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | bash
```

If you already use the Claude Code extension in VS Code, you do not need anything else: rag-kb finds the `claude` program bundled inside the extension (newest version first).

**Sign in once:**

```bash
claude auth login
```

This opens the browser. Sign in with your Claude account; the default (`--claudeai`) uses your subscription. Do not choose `--console`, which bills an API account. Check with `claude auth status`; it should say `"loggedIn": true` and show your plan in `subscriptionType`.

**What rag-kb runs** for each answer:

```
claude -p --output-format json --no-session-persistence --model sonnet --system-prompt-file <temp file> --tools "" --safe-mode --strict-mcp-config
```

- The question and the numbered passages go to the program on **standard input**, never on the command line, so long prompts are safe on Windows.
- The instructions (answer only from the passages, cite with `[n]`, reply `NO_ANSWER` when they do not answer) go in a temporary file that is deleted right after the call.
- `--tools ""`, `--safe-mode` and `--strict-mcp-config` make the call a plain completion with no tools, none of your personal `CLAUDE.md`, hooks, plugins or MCP servers. Each is added only when your version's `claude --help` lists it, so older versions still work.
- The call runs in an empty folder (`<temp>/ragkb-claude-code`) and saves no session.
- The model comes from `RAGKB_CLAUDE_MODEL` (`sonnet`, `opus` or `haiku`), or the choice in the Ask tab.
- `ANTHROPIC_API_KEY` is removed from the environment of the call, so an API key you have set for other work cannot turn your subscription answers into API charges.

**Never add `--bare`.** If you wrap `claude` in your own script or alias, keep `--bare` out of it: in bare mode Claude Code ignores your stored sign-in and accepts only an API key, so every answer would fail with "not signed in".

## Codex (ChatGPT subscription)

**Install:**

```bash
npm install -g @openai/codex
```

The ChatGPT/Codex extension for VS Code also bundles the `codex` program; rag-kb finds it there.

**Sign in once:**

```bash
codex login
```

Choose **Sign in with ChatGPT** in the browser. Check with `codex login status`; it should print `Logged in using ChatGPT`. If it says `Logged in using an API key`, answers are billed to that API account instead of your plan; run `codex logout` and `codex login` again to switch.

**What rag-kb runs** for each answer:

```
codex exec --skip-git-repo-check --ephemeral -s read-only -C <empty temp folder> -o <temp folder>/answer.md --color never -c model_reasoning_effort=medium
```

- Codex has no system-prompt option, so the instructions and the passages travel together on standard input: an "Instructions" section, then a "Task" section with the question and the passages.
- `-s read-only` and the empty temporary folder mean Codex cannot change anything on your machine; it is also told not to run commands or open files. The folder is deleted after the call.
- `--ephemeral` saves no session. The answer is read from `answer.md`.
- `-m <model>` is added when `RAGKB_CODEX_MODEL` is set; empty means the default in your `~/.codex/config.toml`. The model menu lists the model from that file and the models your Codex install has cached.
- `RAGKB_CODEX_REASONING` (`low`, `medium` or `high`, default `medium`) is passed as `model_reasoning_effort`. Lower is faster and uses less of your quota.
- `OPENAI_API_KEY` is removed from the environment of the call.

## Extractive mode (no subscription)

No model at all. The answer is a short digest: a lead line, then one bullet per source with its most relevant sentences and a `[n]` citation. It declines ("not covered") when the passages found are not about the question, so it does not present loosely related text as an answer: each word of the question is weighted by how rare it is among the passages (generic words such as "company" or "year" count little), a specific word that no passage contains declines a short question, and one passage alone must hold at least half of the question's weight. It is always available, it is what `ragkb eval` uses offline, and it is what the MCP tools hand to an assistant that writes the final answer itself (see `CONNECT-YOUR-ASSISTANT.md`).

## Languages

Every provider answers in the language of the question, and the extractive digest quotes the documents as they are. Which passages reach the provider, though, is decided by the search, and the search depends on the **embedding model**, not on the provider.

The default model, `BAAI/bge-small-en-v1.5`, is English-first. When a real share of your documents or questions is in another language, or people ask in one language about documents in another, set `RAGKB_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. That change needs **every document embedded again**: start with a new data folder and upload again, or choose Re-index on every document. The two models have the same size, so rag-kb cannot tell old vectors from new ones; until all documents are re-indexed, search quality is poor. Step by step: "Documents in other languages" in [USER-GUIDE.md](USER-GUIDE.md). A per-collection choice of embedding model is on the [roadmap](ROADMAP.md#2-quality).

A better provider cannot make up for search in the wrong language: if "not covered" comes back for questions in another language that the documents do answer, look at the embedding model first.

## What rag-kb can and cannot see

| rag-kb sees | rag-kb never sees |
|---|---|
| Whether each tool is installed, where, and its version | Your password, login token or API key |
| Signed in or not, and the plan name, from `claude auth status` and `codex login status` | Your other conversations or chat history |
| The answer text, the model name, the time taken and token counts when the tool prints them | Anything in the vendor's credential files |
| The `model` line of `~/.codex/config.toml` and Codex's model list cache, to fill the model menu | |

`claude auth status` also prints your account e-mail and organization; rag-kb reads only the signed-in flag, the sign-in method and the plan, and stores none of it.

**What leaves your machine:** for each question, the question and the passages selected for it, sent by the vendor's own tool to the vendor, exactly as if you had pasted them into Claude or ChatGPT. Your documents as a whole, the index and the embeddings never leave. In extractive mode nothing leaves at all.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| "Claude Code is not installed" or "Codex is not installed" | Install it (above). If it is installed somewhere unusual, set the full path in `RAGKB_CLAUDE_BIN` or `RAGKB_CODEX_BIN`. `ragkb doctor` shows the program it would run and how it found it. |
| "... is not signed in. Run: claude auth login" (or `codex login`) | Run the command shown, finish the browser sign-in, then ask again. The status is re-checked every 30 seconds. |
| **Rate limited** with the vendor's message, for example "You've hit your usage limit" | Your plan's allowance for that tool is used up for now. The message usually says when it resets. Codex on ChatGPT Plus has a fairly small allowance: switch the question to Claude Code or extractive mode, set `RAGKB_PROVIDER=claude_code`, or lower `RAGKB_CODEX_REASONING` to `low`. rag-kb never falls back silently: a rate-limited answer is shown as such. |
| "did not answer within 180 seconds" | The vendor was slow or busy. Ask again, choose a faster model (`haiku`, or lower Codex reasoning), or raise `RAGKB_ANSWER_TIMEOUT_S`. |
| Claude Code answers fail although `claude auth status` says signed in | Check for a wrapper or alias that adds `--bare`, and for `CLAUDE_CODE_USE_BEDROCK` or similar variables that point Claude Code at another account. |
| Answers are billed to an API account | `codex login status` says "API key", or `claude auth status` shows a Console login. Sign out and sign in with the subscription. |
| The first answer of the day is slow | The tools start up and check their login on the first call; later calls are faster. Typical answers take 10 to 30 seconds. |
| "not covered" for something the documents say | The provider worked; search did not find the passage. Read the **Closest passages** under the answer, and see "Reading the states and scores" in `USER-GUIDE.md`. For documents or questions not in English, see "Languages" above. |

Running rag-kb from inside a Claude Code session (for example a terminal opened by Claude Code) works: the variables that mark a nested session are removed before each call.
