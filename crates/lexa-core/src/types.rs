use std::fmt;
use std::path::PathBuf;

use rusqlite::Error as SqlError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub path: String,
    pub mtime: i64,
    pub size: i64,
    pub content_hash: String,
    pub indexed_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: i64,
    pub doc_id: i64,
    pub ord: i64,
    pub byte_start: i64,
    pub byte_end: i64,
    pub line_start: i64,
    pub line_end: i64,
    pub kind: String,
    pub text: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub line_start: i64,
    pub line_end: i64,
    pub score: f32,
    pub excerpt: String,
    pub breakdown: TierBreakdown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierBreakdown {
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub bm25_score: f32,
    pub vector_score: f32,
    pub rerank_score: Option<f32>,
    /// When the caller requested `Auto`, this records the tier we actually
    /// dispatched to so the routing decision is debuggable. `None` if the
    /// caller explicitly named a tier.
    pub routed_to: Option<SearchTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub db_path: PathBuf,
    pub documents: i64,
    pub chunks: i64,
}

/// Retrieval tier — selects the work the search engine does for a query.
///
/// `Instant` and `Dense` are single-retriever ablations (BM25-only and
/// vector-only respectively), exposed mostly for the quality benchmark
/// harness but also useful as product knobs when callers want a pure
/// keyword or pure semantic search. `Fast` and `Deep` are the production
/// hybrid tiers. `Auto` (the new default) routes per-query to the cheapest
/// tier that's likely to do the job, mirroring Exa's `auto` philosophy.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchTier {
    /// FTS5 BM25 only. No model load, no vector index access.
    Instant,
    /// Vector index only (binary-quantized Hamming KNN). Uses the dense
    /// embedder; primarily an ablation for quality benchmarks.
    Dense,
    /// BM25 + dense + RRF fusion. Production hybrid tier.
    Fast,
    /// `Fast` + cross-encoder reranking over the top-K fused candidates.
    Deep,
    /// Inspect the query and route to the cheapest tier likely to answer it
    /// well: identifier-shaped short queries → `Instant`; long question-form
    /// queries (or `[deep]` prefix) → `Deep`; everything else → `Fast`.
    /// Default tier.
    #[default]
    Auto,
}

impl std::str::FromStr for SearchTier {
    type Err = LexaError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "instant" | "bm25" => Ok(Self::Instant),
            "dense" | "vector" => Ok(Self::Dense),
            "fast" | "hybrid" => Ok(Self::Fast),
            "deep" => Ok(Self::Deep),
            "auto" => Ok(Self::Auto),
            other => Err(LexaError::InvalidTier(other.to_string())),
        }
    }
}

impl fmt::Display for SearchTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Instant => "instant",
            Self::Dense => "dense",
            Self::Fast => "fast",
            Self::Deep => "deep",
            Self::Auto => "auto",
        };
        f.write_str(name)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LexaError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Sql(#[from] SqlError),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("pdf extraction error: {0}")]
    Pdf(String),
    #[error("invalid tier '{0}', expected instant, fast, or deep")]
    InvalidTier(String),
    #[error("invalid path: {0}")]
    InvalidPath(String),
    #[error("unsupported benchmark: {0}")]
    UnsupportedBenchmark(String),
}
