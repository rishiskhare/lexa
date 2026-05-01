//! Obsidian-vault adapter for [Lexa](https://github.com/rishiskhare/lexa).
//!
//! Wraps `lexa-core`'s `LexaDb` with vault-aware indexing — frontmatter
//! parsing, wiki-link extraction, tag indexing, block-id tracking — and
//! exposes a note-shaped API:
//!
//! - `LexaObsidianDb::index_vault` — walk a vault directory, parse
//!   frontmatter, run `LexaDb::index_path`, populate sidecar tables in
//!   the same SQLite file.
//! - `LexaObsidianDb::search_notes` — hybrid search with tag / folder
//!   filters layered on top of the lexa-core hybrid retrieval.
//! - `LexaObsidianDb::find_backlinks` / `list_tags` / `get_note` /
//!   `get_similar` — note-aware queries against the sidecar tables.
//!
//! Companion binaries:
//!
//! - `lexa-obsidian` — CLI: `index`, `status`, `tags`, `backlinks`,
//!   `search`, `watch`.
//! - `lexa-obsidian-mcp` — rmcp stdio MCP server for Codex / Claude
//!   Desktop / any MCP client.

pub mod db;
pub mod frontmatter;
pub mod schema;
pub mod tags;
pub mod wikilinks;

pub use lexa_core::LexaError;

pub type Result<T> = std::result::Result<T, lexa_core::LexaError>;

pub use db::{
    Backlink, IndexReport, LexaObsidianDb, LinkRef, Note, NoteHit, SearchNotesOptions, TagCount,
    VaultStatus,
};
pub use frontmatter::Frontmatter;
