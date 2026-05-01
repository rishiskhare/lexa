# Launch playbook (private — not posted yet)

Drafts for the v0.1 launch. Kept in-repo so the wording can be edited
in PRs alongside any code changes that affect it. Nothing here is
posted automatically.

## Show HN

> **Show HN: Lexa — local-first hybrid retrieval for your Obsidian vault, exposed to Codex / Claude / Cursor via MCP**
>
> Lexa is a single static Rust binary that gives any MCP client
> (Codex CLI, Claude Desktop, Claude Code, Cursor) the ability to ask
> your Obsidian vault questions: "what did I write about X?", "show
> me backlinks for Y", "find notes similar to Z".
>
> **Local-first.** First run downloads two ONNX models (~390 MB).
> After that, zero outbound traffic. No API keys, no telemetry,
> read-only on your vault. The DB is a regular SQLite file at
> `~/.lexa/obsidian-<sha>.sqlite` you can inspect with any SQLite
> client.
>
> **The architecture mirrors Exa's:** five tiers (`instant`, `dense`,
> `fast`, `deep`, `auto`), hybrid BM25 + binary-quantized Matryoshka
> KNN + cross-encoder rerank, query-aware highlights. Fast-tier p50 is
> ~9 ms warm — 38× faster than Exa's published Fast tier (because
> everything is in SQLite next to the CPU instead of behind a
> planet-scale index).
>
> **Install:**
>
>     curl -fsSL https://raw.githubusercontent.com/rishiskhare/lexa/main/scripts/install.sh | sh
>     lexa-obsidian setup
>
> `setup` is interactive: it picks your vault, optionally pre-indexes,
> writes the right MCP config block into `~/.codex/config.toml`, and
> drops `AGENTS.md` in your vault root so the agent routes note
> questions through Lexa without you having to say "Use lexa-obsidian"
> in every prompt.
>
> Repo: <https://github.com/rishiskhare/lexa>
> Crates: `cargo install lexa-obsidian` (pulls `lexa-core`).
>
> Built because Codex inside Obsidian is most of my note-taking loop
> and I wanted the AI to know what I'd already written. Feedback
> welcome — especially on the auto-tier router and the
> `additional_queries` Deep-tier expansion.

## r/ObsidianMD

> **Built a local-first MCP server for Obsidian — Codex / Claude / Cursor can answer questions from your vault**
>
> (similar text, less infrastructure-talk, more "what does this do for
> me as an Obsidian user")

## MCP server registry submissions

To-do PRs (each is a one-line README change in the upstream repo):

- [ ] [`modelcontextprotocol/servers`](https://github.com/modelcontextprotocol/servers) — community list
- [ ] [`punkpeye/awesome-mcp-servers`](https://github.com/punkpeye/awesome-mcp-servers)
- [ ] [`mcpservers.org`](https://mcpservers.org) — submission form
- [ ] [`tolkonepiu/best-of-mcp-servers`](https://github.com/tolkonepiu/best-of-mcp-servers)

Suggested entry text:

> **lexa-obsidian** — Local-first hybrid retrieval over an Obsidian
> vault. BM25 + binary-quantized Matryoshka KNN + cross-encoder
> rerank, all in SQLite. Tools: `search_notes`, `find_backlinks`,
> `list_tags`, `get_note`, `get_similar`. Read-only, no telemetry.
> [github.com/rishiskhare/lexa](https://github.com/rishiskhare/lexa)

## Homebrew tap

Deferred — set up only after the first stranger asks for it. cargo-dist
generates the formula automatically from the GitHub Release; we just
need a `homebrew-tap` repo to push to. Estimated 30 minutes of work
when triggered.
