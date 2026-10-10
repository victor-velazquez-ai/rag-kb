# Connect your assistant

While `ragkb serve` runs, the knowledge base is also an **MCP server** at:

```
http://127.0.0.1:8765/mcp
```

MCP (Model Context Protocol) is the standard way assistants call outside tools. Once you connect Claude Code, Codex, Claude Desktop or claude.ai to this address, you can ask your questions there and your assistant looks them up in your documents. The assistant writes the answer on **your own subscription**; rag-kb never calls a model for these tools, it only searches and hands back the passages.

If you changed `RAGKB_PORT`, use that port in every address below. `ragkb serve` prints the exact address on start (`MCP endpoint: http://127.0.0.1:8765/mcp`).

## The tools

| Tool | Arguments | Returns |
|---|---|---|
| `list_collections` | none | Every collection: `id`, `name`, `description`, `documents` (how many are searchable). Assistants call this first. |
| `list_documents` | `collection_id` | The collection's documents: `id`, `title`, `filename`, `status`, `pages`, `chunks`, `indexed_at`. |
| `search_knowledge_base` | `collection_id`, `query`, `top_k` (1 to 20, default 6) | The best passages in rank order: `title`, `section`, `page`, `text`, `score`. No answer is written. |
| `ask_knowledge_base` | `collection_id`, `question`, `top_k` (1 to 20, default 6) | A brief for the assistant (below). |

`collection_id` also accepts a collection's exact name, so "use the HR collection" works without looking up the id.

All four tools only read. Uploading, re-indexing and deleting stay in the rag-kb web page.

### What `ask_knowledge_base` returns

The same answer contract the web page uses, produced with the built-in extractive mode, plus two extra fields:

| Field | Meaning |
|---|---|
| `status` | `answered`, `no_answer` (the documents do not cover it), `no_documents` (the collection is empty) or `retrieval_unavailable`. |
| `answer` | A digest of the best sentences with `[n]` markers, or `null` when not answered. |
| `message` | One plain sentence when not answered. |
| `citations` | Numbered sources 1..K: `marker`, `title`, `page`, `section`, `snippet`, `score`. |
| `passages` | The full text of each cited passage, under the same `marker`. |
| `guidance` | Two or three sentences telling the assistant how to write the final answer: in the user's language, direct answer first, inline `[n]` citations that match the sources, and a plain "the documents do not cover this" when they do not. |
| `knowledge_version`, `id`, `created_at`, `language`, `passages_considered`, `provider` | Bookkeeping. The `id` lets you find the same brief later in the app. |

### Errors

A tool that cannot do what was asked returns `{"error": {"code": "...", "message": "..."}}` and marks the result as an error. Codes: `not_found` (unknown collection), `invalid_request` (empty question, `top_k` out of range, unknown argument), `unknown_tool`, `retrieval_unavailable`, `unavailable` (a part of the app is not running) and `internal_error` (details only in the server log). Messages are plain sentences and never contain file paths or stack traces.

## Claude Code

```bash
claude mcp add --transport http rag-kb http://127.0.0.1:8765/mcp
```

Add `--scope user` to have it in every project instead of only the current folder. Check with `claude mcp list` (it should show `rag-kb ... Connected`), or type `/mcp` inside a Claude Code session. Then ask, for example: "Use rag-kb to find out how many days of annual leave we get."

## Codex

```bash
codex mcp add rag-kb --url http://127.0.0.1:8765/mcp
```

This writes the following to `~/.codex/config.toml` (on Windows `%USERPROFILE%\.codex\config.toml`), which you can also add by hand:

```toml
[mcp_servers.rag-kb]
url = "http://127.0.0.1:8765/mcp"
```

Check with `codex mcp list`. Codex may print "MCP server may or may not require login"; rag-kb needs no login on your own machine, so ignore it.

## Claude Desktop

Claude Desktop's configuration file only starts local programs; it cannot point at a web address directly. The standard bridge is `mcp-remote`, a small program that runs with Node.js 18 or newer (`node --version` to check; install from nodejs.org if needed).

1. In Claude Desktop open **Settings > Developer > Edit Config**. This opens `claude_desktop_config.json`:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add the `rag-kb` entry (keep any servers already listed):

   ```json
   {
     "mcpServers": {
       "rag-kb": {
         "command": "npx",
         "args": ["-y", "mcp-remote", "http://127.0.0.1:8765/mcp"]
       }
     }
   }
   ```

3. Quit Claude Desktop completely (from the tray icon on Windows or the menu bar on macOS, not only the window) and start it again.
4. Start `ragkb serve` before you ask. The tools appear in the chat's tools menu under `rag-kb`.

On Windows, if Claude Desktop reports that it cannot start `npx`, use the full path instead: `"command": "C:\\Program Files\\nodejs\\npx.cmd"`. The first start downloads `mcp-remote` (verified with version 0.14.3), which takes a few seconds.

## claude.ai in the browser (custom connector)

claude.ai connects to MCP servers **from Anthropic's cloud**, not from your browser. `127.0.0.1` on your laptop is therefore unreachable for it; it needs a public HTTPS address. On a laptop that means a tunnel. The same applies to custom connectors added through Claude Desktop's **Connectors** settings (as opposed to the config file above).

1. Start rag-kb with the MCP guard opened for the tunnel's address. Keep `RAGKB_HOST` at `127.0.0.1`; the tunnel connects locally.

   ```bash
   # macOS / Linux
   RAGKB_MCP_PUBLIC=true ragkb serve
   ```
   ```powershell
   # Windows PowerShell
   $env:RAGKB_MCP_PUBLIC = "true"; ragkb serve
   ```

2. In a second terminal, start a tunnel to port 8765. Either of these prints a public `https://` address:

   ```bash
   cloudflared tunnel --url http://127.0.0.1:8765     # prints https://<random words>.trycloudflare.com
   ngrok http 8765                                     # prints https://<random>.ngrok-free.app
   ```

3. In claude.ai open **Customize > Connectors** (on some plans **Settings > Connectors**), choose **Add**, then **Add custom connector**.
4. Name it `rag-kb`, paste the tunnel address with `/mcp` at the end (for example `https://<random words>.trycloudflare.com/mcp`), and continue.
5. For authentication choose **No sign in**, then **Add**.
6. In a chat, turn the `rag-kb` connector on from the tools menu and ask your question.

Custom connectors work on the Free, Pro, Max, Team and Enterprise plans (on Team and Enterprise an owner may have to allow them). ChatGPT can use the same public `/mcp` address through its custom connectors (developer mode) on Plus and Pro.

**Before you do this, understand what the tunnel exposes.** In this version there is no sign-in. While the tunnel runs, anyone who has the address can search your documents through `/mcp`. The tunnel forwards the whole app, but the API that lists, uploads and deletes documents (`/api`) still answers only requests addressed to this machine, so through the tunnel the web page loads without data and every `/api` call except `GET /api/health` gets HTTP 403. Leave `RAGKB_API_PUBLIC` at `false` while a tunnel runs. Quick tunnel addresses are long and random, which helps, but they are not a password. Run the tunnel only while you use it, stop it (Ctrl+C) when you are done, and do not use this with documents you could not afford to leak. Objective 2 adds a proper sign-in (OAuth, the connector's "Sign in" option) before any public deployment.

## How the loopback guard works

By default (`RAGKB_MCP_PUBLIC=false`) the `/mcp` endpoint answers only requests addressed to this machine: the `Host` header must be `localhost`, `127.x.x.x` or `[::1]`, with any port. Requests sent from a web page on another site (a browser `Origin` that is not local) are refused as well, which blocks DNS-rebinding tricks. A refused request gets HTTP 403 with:

```json
{"error": {"code": "forbidden", "message": "The MCP endpoint only answers on this machine (localhost). ..."}}
```

Setting `RAGKB_MCP_PUBLIC=true` turns this check off so a tunnel or reverse proxy can reach `/mcp`. It does not change where the server listens; that is `RAGKB_HOST`.

The API at `/api` has the same guard on the `Host` header, controlled separately by `RAGKB_API_PUBLIC` (default `false`), so opening `/mcp` for an assistant never opens the API. `GET /api/health` always answers. Set `RAGKB_API_PUBLIC=true` only when the server sits behind your own sign-in, for example to use the web page from another machine.

Both guards trust the `Host` header. A tunnel or proxy that rewrites `Host` to `localhost` (for example ngrok with `--host-header=rewrite`) makes remote requests look local and gets past them, so do not use that option with rag-kb.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The assistant does not list the rag-kb tools | `ragkb serve` is not running, or the port differs. Start it, check the address it prints, restart the assistant (Claude Desktop needs a full quit). |
| HTTP 403 `forbidden` from `/mcp` through a tunnel | The guard is on. Restart with `RAGKB_MCP_PUBLIC=true`. |
| HTTP 403 `forbidden` from `/api`, or an empty web page, through a tunnel | Intended: the API answers only on this machine (`RAGKB_API_PUBLIC=false`). Use the web page at `http://127.0.0.1:8765`. |
| HTTP 503 `not_ready` | The app is still starting. Try again in a second. |
| claude.ai says it cannot reach the server | A `127.0.0.1` address was used, the tunnel stopped, or `/mcp` is missing from the end of the address. |
| Every answer comes back `no_documents` | The collection is empty or its documents are still processing. Check with `list_documents` or the Documents tab. |
| `no_answer` for something the documents do say | Search did not find the passage. Try `search_knowledge_base` with different words, raise `top_k`, or check that the document indexed (status `ready`). |

To check the endpoint by hand:

```bash
curl -s -X POST http://127.0.0.1:8765/mcp -H "content-type: application/json" -H "accept: application/json, text/event-stream" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"list_collections\",\"arguments\":{}}}"
```

A working server answers with a JSON result that lists your collections.

## Technical notes

- Transport: MCP streamable HTTP, stateless, JSON responses (no server-sent event streams), so it works through tunnels and proxies and with clients on both the 2025 handshake and the newer protocol versions. Both `/mcp` and `/mcp/` work.
- Every `ask_knowledge_base` call is stored like an answer from the web page and written to the audit log (question length, status, timings; never the passages).
- `RAGKB_MCP_ENABLED=false` removes the endpoint entirely.
