# lexa-obsidian

> Local-first hybrid retrieval over an Obsidian vault, exposed to Codex
> / Claude Desktop / Cursor / Claude Code via MCP. Part of the
> [Lexa](https://github.com/rishiskhare/lexa) workspace.

```bash
curl -fsSL https://raw.githubusercontent.com/rishiskhare/lexa/main/scripts/install.sh | sh
lexa-obsidian setup
```

`setup` is interactive: it asks for your vault path, optionally pre-
indexes (recommended for >1 000-note vaults), writes the right MCP
config block into `~/.codex/config.toml` (and Claude Desktop / Claude
Code if you opt in), and drops an `AGENTS.md` in your vault root so
agents route note questions through Lexa without an explicit prefix.

Restart your MCP client and ask, e.g.:

```text
> what did I write about <topic>?
> list my top 10 tags
> show me backlinks for "<note name>"
```

## What's in the box

Two binaries:

| Binary               | What it does                                                              |
|----------------------|---------------------------------------------------------------------------|
| `lexa-obsidian`      | CLI: `setup`, `doctor`, `index`, `status`, `tags`, `backlinks`, `search`, `watch`, `models prefetch`. |
| `lexa-obsidian-mcp`  | rmcp stdio server. Tools: `search_notes`, `find_backlinks`, `list_tags`, `get_note`, `get_similar`, `index_vault`, `purge_vault`, `vault_status`. |

Plus a library API (`LexaObsidianDb::open`, `index_vault`, `search_notes`,
…) for callers who want to embed Lexa's vault retrieval into their own
Rust app.

## How it works

Built on [`lexa-core`](https://crates.io/crates/lexa-core)'s hybrid
retrieval (BM25 + binary-quantized 768-d Matryoshka KNN +
cross-encoder rerank). The Obsidian-specific layer adds:

- **Frontmatter stripped before embedding** so `title:`, `tags:`,
  `aliases:`, and custom YAML fields don't pollute the vector
  representation. The full frontmatter is preserved in a sidecar
  `note_metadata` table.
- **Wiki-link graph**: `[[Note]]`, `[[Note|Alias]]`, `[[Note#Header]]`,
  `[[Note^block]]`, `![[Embed]]` are parsed into a `note_links` table.
  Backlinks are a single SQL JOIN.
- **Tag index**: frontmatter `tags:` plus inline `#tag` (including
  nested `#project/lexa`), lowercase-normalised. Code fences and
  heading lines are correctly skipped.
- **Block addressing**: trailing `^block-id` markers persist into
  `note_blocks` and are queryable via `get_note { block: "^abc" }`.
- **Per-vault DB**: `~/.lexa/obsidian-<sha-of-vault>.sqlite` so two
  distinct vaults never share an index.

## Privacy

- 100 % local. First run downloads two ONNX models (~390 MB total)
  from Hugging Face; nothing leaves your machine after that.
- Read-only on your vault — the MCP server cannot create, edit, or
  delete notes.
- No telemetry, no analytics, no API keys.

## Links

- Repo: <https://github.com/rishiskhare/lexa>
- Architecture: [docs/ARCHITECTURE.md](https://github.com/rishiskhare/lexa/blob/main/docs/ARCHITECTURE.md)
- Decision log: [docs/adr/006-obsidian.md](https://github.com/rishiskhare/lexa/blob/main/docs/adr/006-obsidian.md)
- FAQ: [docs/FAQ.md](https://github.com/rishiskhare/lexa/blob/main/docs/FAQ.md)
- Benchmarks: [docs/BENCHMARKS.md](https://github.com/rishiskhare/lexa/blob/main/docs/BENCHMARKS.md)

Dual-licensed MIT OR Apache-2.0.
