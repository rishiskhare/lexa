mod chunk;
mod db;
mod embed;
mod query;
mod search;
mod types;

pub use db::{default_db_path, open, LexaDb};
pub use embed::{matryoshka_truncate, EmbeddingBackend, EmbeddingConfig, EMBEDDING_DIMS};
pub use search::SearchOptions;
pub use types::{Chunk, Document, IndexStats, LexaError, SearchHit, SearchTier, TierBreakdown};

pub type Result<T> = std::result::Result<T, LexaError>;
