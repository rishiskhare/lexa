# ADR-006: Obsidian adapter (`lexa-obsidian`)

## Status

Accepted, shipped 2026-05-01.

## Context

Lexa's `lexa-mcp` server is file-shaped (`search_files`, `index_path`),
which is right for code corpora but the wrong vocabulary for Obsidian.
Obsidian users think in **notes** (titles, aliases, tags), **links**
(`[[wikilinks]]` / `![[embeds]]` / backlinks) and **blocks** (`^id`
references). Three concrete problems blocked the use case before this
ADR:

1. The markdown chunker treats `---` frontmatter as ordinary text, so
   YAML metadata leaked into the first chunk's embedding and dragged
   vector relevance down.
2. There was no link graph, no tag index, no block addressing — the
   queries Obsidian-native users actually make were impossible.
3. The MCP tools didn't speak the right shape: `search_files` returns
   chunk excerpts but no note title, no tag set, no link graph context.

## Decisions

1. **New crate, not a feature flag.** `crates/lexa-obsidian/` lives in
   the same workspace, path-depends on `lexa-core`, and ships two
   binaries: `lexa-obsidian` (CLI) and `lexa-obsidian-mcp` (rmcp stdio
   server). A profile flag on `lexa-mcp` would have polluted the
   markdown chunker and the rmcp tool router with vault-aware
   conditional logic; the new crate keeps the engine generic.

2. **Sidecar tables, not a JSON column on `documents`.** Frontmatter,
   tags, links, and block-ids live in four new tables:

   - `note_metadata(doc_id PK, title, aliases_json, raw_json)`
   - `note_links(id PK, src_doc_id, target_name, target_path, header, block_id, alias, kind)`
   - `note_tags(doc_id, tag, PK(doc_id, tag))`
   - `note_blocks(chunk_id PK, doc_id, block_id, UNIQUE(doc_id, block_id))`

   All ride on `ON DELETE CASCADE` against `documents.id`, so purging a
   note through the core API cleans the sidecars automatically. The
   migration runs idempotently on `LexaObsidianDb::open` against the
   same SQLite file as `lexa-core`'s schema; running `lexa-mcp` and
   `lexa-obsidian-mcp` against the same DB is harmless because the
   former simply ignores the sidecar tables.

3. **Preprocessor hook on `lexa-core`.** Rather than re-walk the vault
   after `index_path`, `LexaDb::index_path_with_preprocessor` lets a
   caller substitute the text used for chunking and supply a
   caller-defined `payload`. The Obsidian crate's `ObsidianPreprocessor`
   strips frontmatter and parses tags + links eagerly; the matching
   `commit_sidecar` callback writes `note_metadata`, `note_tags`,
   `note_links` **inside the same transaction** as the chunk insert.
   Block ids are populated in a follow-up pass because they need
   `chunks.id`, which only exists post-commit.

4. **`SearchHit.heading` exposed in `lexa-core`.** The chunker already
   captured the parent heading in `chunks.context`; `SearchHit` now
   surfaces it (added field, defaults to `None`). The MCP `search_notes`
   tool returns it under `heading`. Non-breaking JSON addition.

5. **Lazy indexing with explicit fallback.** The MCP server's
   `vault_status` flips `needs_index = true` on a fresh DB; the next
   content-bearing tool call (`search_notes`, `find_backlinks`,
   `get_note`, `get_similar`, `list_tags`) triggers a synchronous
   `index_vault` inside the call. For >1000-note vaults, the user is
   expected to run `lexa-obsidian index` ahead of time so the first MCP
   call returns immediately.

6. **Per-vault DB path by default.** `lexa-obsidian-mcp` derives
   `~/.lexa/obsidian-<sha-of-vault>.sqlite` from the canonical vault
   path. Two vaults can never share an index, and the user can override
   with `LEXA_OBSIDIAN_DB`.

7. **Same Nomic v1.5-Q embedder as the core engine.** No second model
   download. Same schema (`vectors_bin bit[768]` +
   `vectors_bin_preview bit[256]`). Two-stage Matryoshka KNN works
   identically.

## Consequences

- Codex / Claude Desktop / Cursor users can wire one MCP server entry
  and immediately ask their notes questions like "what did I write
  about latency budgets last quarter?" with backlink context.
- The sidecar schema is independently inspectable with `sqlite3` (or
  any SQLite browser); each table is small (~one row per note for
  `note_metadata` and `note_tags`, one row per link for `note_links`).
- The crate is independently publishable: `lexa-core` + `lexa-obsidian`
  is a coherent retrieval-engine + Obsidian-frontend pair with no
  coupling to the code-search-flavoured `lexa-cli` / `lexa-mcp`.

## What's deferred

- Bidirectional sync (writing notes through Lexa). Out of scope; users
  compose `lexa-obsidian` (read) with the existing
  `MarkusPfundstein/mcp-obsidian` (write) for the full loop.
- Embed inlining: `![[Note]]` is stored as an edge but not inlined into
  the parent's embedding text. A planned phase-2 feature.
- Aliases as standalone search hits (currently stored in
  `note_metadata.aliases_json` but not separately indexed).
- Dataview blocks; canvas / Excalidraw files.
