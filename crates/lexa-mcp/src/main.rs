use std::path::PathBuf;

use lexa_core::{
    default_db_path, open, EmbeddingBackend, EmbeddingConfig, SearchOptions, SearchTier,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt,
};
use serde::Deserialize;

#[derive(Debug, Clone)]
struct LexaServer {
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchFilesRequest {
    #[schemars(description = "Natural language or keyword query")]
    query: String,
    #[schemars(description = "Retrieval tier: instant, fast, or deep")]
    tier: Option<String>,
    #[schemars(description = "Maximum number of results")]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PathRequest {
    #[schemars(description = "File or directory path")]
    path: String,
}

impl LexaServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl LexaServer {
    #[tool(description = "Search indexed local files")]
    fn search_files(
        &self,
        Parameters(SearchFilesRequest { query, tier, limit }): Parameters<SearchFilesRequest>,
    ) -> Result<String, ErrorData> {
        let tier = tier
            .as_deref()
            .unwrap_or("fast")
            .parse::<SearchTier>()
            .map_err(|error| invalid_params(error.to_string()))?;
        let db = db()?;
        let hits = db
            .search(&SearchOptions {
                query,
                tier,
                limit: limit.unwrap_or(10),
                additional_queries: Vec::new(),
            })
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&hits).map_err(internal_error)
    }

    #[tool(description = "Index a file or directory")]
    fn index_path(
        &self,
        Parameters(PathRequest { path }): Parameters<PathRequest>,
    ) -> Result<String, ErrorData> {
        let mut db = db()?;
        let count = db.index_path(path).map_err(internal_error)?;
        Ok(format!("indexed {count} file(s)"))
    }

    #[tool(description = "List indexed paths")]
    fn list_indexed_paths(&self) -> Result<String, ErrorData> {
        let db = db()?;
        let docs = db.list_documents().map_err(internal_error)?;
        serde_json::to_string_pretty(&docs).map_err(internal_error)
    }

    #[tool(description = "Remove an indexed path")]
    fn purge_path(
        &self,
        Parameters(PathRequest { path }): Parameters<PathRequest>,
    ) -> Result<String, ErrorData> {
        let mut db = db()?;
        let count = db.purge_path(path).map_err(internal_error)?;
        Ok(format!("purged {count} file(s)"))
    }

    #[tool(description = "Return index status")]
    fn status(&self) -> Result<String, ErrorData> {
        let db = db()?;
        let stats = db.stats().map_err(internal_error)?;
        serde_json::to_string_pretty(&stats).map_err(internal_error)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LexaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Lexa local file search over the configured SQLite index")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    LexaServer::new()
        .serve(rmcp::transport::stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}

fn db() -> Result<lexa_core::LexaDb, ErrorData> {
    let db_path = std::env::var_os("LEXA_DB")
        .map(PathBuf::from)
        .unwrap_or_else(default_db_path);
    let backend = match std::env::var("LEXA_EMBEDDER").ok().as_deref() {
        Some("hash") => EmbeddingBackend::Hash,
        _ => EmbeddingBackend::FastEmbed,
    };
    let config = EmbeddingConfig {
        backend,
        show_download_progress: false,
    };
    open(db_path, config).map_err(internal_error)
}

fn invalid_params(message: impl Into<String>) -> ErrorData {
    ErrorData::invalid_params(message.into(), None)
}

fn internal_error(error: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(error.to_string(), None)
}
