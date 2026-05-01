//! `lexa-obsidian-mcp` — rmcp stdio MCP server for Obsidian-aware retrieval.
//!
//! Configure in `~/.codex/config.toml`:
//!
//! ```toml
//! [mcp_servers.lexa-obsidian]
//! command = "lexa-obsidian-mcp"
//! env = { LEXA_OBSIDIAN_VAULT = "/Users/<you>/Documents/MyVault" }
//! ```
//!
//! Tools (mirroring Exa's content-API and search-API surface, adapted
//! for Obsidian):
//!
//! - `search_notes` — hybrid retrieval; supports tag and folder filters.
//! - `find_backlinks` — list notes linking to a given note.
//! - `list_tags` — top tags by usage.
//! - `get_note` — full note (frontmatter + body + outgoing/incoming).
//! - `get_similar` — notes semantically similar to a given note.
//! - `index_vault` — explicit re-index.
//! - `purge_vault` — drop the index.
//! - `vault_status` — DB + sidecar counts plus `needs_index` flag.
//!
//! Indexing behaviour: lazy on the **first call** that requires data.
//! If `vault_status().needs_index` is true on startup, the next
//! `search_notes` / `get_note` / `get_similar` / `find_backlinks` call
//! triggers a synchronous `index_vault` so callers don't see stale
//! responses. Run `lexa-obsidian index` ahead of time on large vaults
//! to avoid the wait.

use std::path::PathBuf;
use std::sync::Mutex;

use lexa_core::{EmbeddingBackend, EmbeddingConfig, SearchTier};
use lexa_obsidian::{LexaObsidianDb, SearchNotesOptions};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt,
};
use serde::Deserialize;

struct LexaObsidianServer {
    tool_router: ToolRouter<Self>,
    db: Mutex<LexaObsidianDb>,
    indexed: Mutex<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchNotesRequest {
    #[schemars(description = "Natural language or keyword query")]
    query: String,
    #[schemars(description = "Retrieval tier: instant, dense, fast, deep, auto (default)")]
    tier: Option<String>,
    #[schemars(description = "Maximum number of results")]
    limit: Option<usize>,
    #[schemars(description = "Filter to notes that carry any of these tags")]
    tags: Option<Vec<String>>,
    #[schemars(
        description = "Filter to notes whose path begins with one of these folders (relative to the vault root)"
    )]
    folders: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct NoteRequest {
    #[schemars(description = "Path or filename stem of the note")]
    note: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetNoteRequest {
    #[schemars(description = "Path or filename stem of the note")]
    note: String,
    #[schemars(description = "Optional Obsidian block id (e.g. ^abc-123)")]
    block: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetSimilarRequest {
    #[schemars(description = "Path or filename stem of the seed note")]
    note: String,
    #[schemars(description = "Maximum number of similar notes to return")]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListTagsRequest {
    #[schemars(description = "Optional case-insensitive prefix filter")]
    prefix: Option<String>,
    #[schemars(description = "Maximum number of tags to return (default 50)")]
    limit: Option<usize>,
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
#[serde(default)]
struct EmptyRequest {}

impl LexaObsidianServer {
    fn new(db: LexaObsidianDb) -> Self {
        let already_indexed = db.vault_status().map(|s| !s.needs_index).unwrap_or(false);
        Self {
            tool_router: Self::tool_router(),
            db: Mutex::new(db),
            indexed: Mutex::new(already_indexed),
        }
    }

    /// Make sure the vault has been indexed at least once before serving
    /// content-bearing requests. Idempotent on subsequent calls thanks
    /// to lexa-core's content-hash skip.
    fn ensure_indexed(&self) -> Result<(), ErrorData> {
        let mut flag = self
            .indexed
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        if *flag {
            return Ok(());
        }
        let mut db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        eprintln!("lexa-obsidian: priming index (one-time)…");
        db.index_vault().map_err(internal_error)?;
        *flag = true;
        Ok(())
    }
}

#[tool_router]
impl LexaObsidianServer {
    #[tool(description = "Hybrid search over indexed Obsidian notes")]
    fn search_notes(
        &self,
        Parameters(req): Parameters<SearchNotesRequest>,
    ) -> Result<String, ErrorData> {
        self.ensure_indexed()?;
        let tier: SearchTier = req
            .tier
            .as_deref()
            .unwrap_or("auto")
            .parse()
            .map_err(|err: lexa_core::LexaError| invalid_params(err.to_string()))?;
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let hits = db
            .search_notes(&SearchNotesOptions {
                query: req.query,
                tier,
                limit: req.limit.unwrap_or(10),
                tags: req.tags.unwrap_or_default(),
                folders: req.folders.unwrap_or_default(),
                additional_queries: Vec::new(),
            })
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&hits).map_err(internal_error)
    }

    #[tool(description = "List notes linking to a given note (path or filename stem)")]
    fn find_backlinks(
        &self,
        Parameters(req): Parameters<NoteRequest>,
    ) -> Result<String, ErrorData> {
        self.ensure_indexed()?;
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let backlinks = db.find_backlinks(&req.note).map_err(internal_error)?;
        serde_json::to_string_pretty(&backlinks).map_err(internal_error)
    }

    #[tool(description = "List the most-used tags in the vault")]
    fn list_tags(&self, Parameters(req): Parameters<ListTagsRequest>) -> Result<String, ErrorData> {
        self.ensure_indexed()?;
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let tags = db
            .list_tags(req.prefix.as_deref(), req.limit.unwrap_or(50))
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&tags).map_err(internal_error)
    }

    #[tool(
        description = "Fetch a single note (frontmatter, body, outgoing + incoming links, tags). Optionally restricted to a block id."
    )]
    fn get_note(&self, Parameters(req): Parameters<GetNoteRequest>) -> Result<String, ErrorData> {
        self.ensure_indexed()?;
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let note = db
            .get_note(&req.note, req.block.as_deref())
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&note).map_err(internal_error)
    }

    #[tool(description = "Find notes semantically similar to the given note")]
    fn get_similar(
        &self,
        Parameters(req): Parameters<GetSimilarRequest>,
    ) -> Result<String, ErrorData> {
        self.ensure_indexed()?;
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let hits = db
            .get_similar(&req.note, req.limit.unwrap_or(10))
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&hits).map_err(internal_error)
    }

    #[tool(description = "Force a full re-index of the configured vault")]
    fn index_vault(&self, Parameters(_req): Parameters<EmptyRequest>) -> Result<String, ErrorData> {
        let mut db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let report = db.index_vault().map_err(internal_error)?;
        let mut flag = self
            .indexed
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        *flag = true;
        serde_json::to_string_pretty(&report).map_err(internal_error)
    }

    #[tool(description = "Drop every indexed note for the configured vault")]
    fn purge_vault(&self, Parameters(_req): Parameters<EmptyRequest>) -> Result<String, ErrorData> {
        let mut db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let count = db.purge_vault().map_err(internal_error)?;
        let mut flag = self
            .indexed
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        *flag = false;
        Ok(format!("purged {count} note(s)"))
    }

    #[tool(description = "Return DB + sidecar counts and the `needs_index` flag")]
    fn vault_status(
        &self,
        Parameters(_req): Parameters<EmptyRequest>,
    ) -> Result<String, ErrorData> {
        let db = self
            .db
            .lock()
            .map_err(|err| internal_error(err.to_string()))?;
        let status = db.vault_status().map_err(internal_error)?;
        serde_json::to_string_pretty(&status).map_err(internal_error)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LexaObsidianServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Lexa Obsidian: hybrid (BM25 + dense + rerank) retrieval over the configured vault.",
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = open_db_from_env()?;
    LexaObsidianServer::new(db)
        .serve(rmcp::transport::stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}

fn open_db_from_env() -> anyhow::Result<LexaObsidianDb> {
    let vault = std::env::var_os("LEXA_OBSIDIAN_VAULT")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("LEXA_OBSIDIAN_VAULT must be set to the vault root path"))?;
    let canonical = std::fs::canonicalize(&vault)
        .map_err(|err| anyhow::anyhow!("vault path is not a directory: {err}"))?;
    let db_path = std::env::var_os("LEXA_OBSIDIAN_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_obsidian_db_path(&canonical));
    let backend = match std::env::var("LEXA_EMBEDDER").ok().as_deref() {
        Some("hash") => EmbeddingBackend::Hash,
        _ => EmbeddingBackend::FastEmbed,
    };
    let config = EmbeddingConfig {
        backend,
        show_download_progress: false,
    };
    Ok(LexaObsidianDb::open(db_path, canonical, config)?)
}

fn default_obsidian_db_path(vault: &std::path::Path) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    vault.to_string_lossy().hash(&mut hasher);
    let dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lexa");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("obsidian-{:016x}.sqlite", hasher.finish()))
}

fn invalid_params(message: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(message.into(), None)
}

fn internal_error(error: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(error.to_string(), None)
}
