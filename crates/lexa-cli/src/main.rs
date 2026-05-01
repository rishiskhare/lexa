use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use lexa_core::{
    default_db_path, open, EmbeddingBackend, EmbeddingConfig, SearchOptions, SearchTier,
};
use notify::{RecursiveMode, Watcher};

#[derive(Debug, Parser)]
#[command(
    name = "lexa",
    version,
    about = "Local hybrid search for files and code"
)]
struct Cli {
    #[arg(long, global = true)]
    db: Option<PathBuf>,
    #[arg(long, global = true, hide = true)]
    hash_embeddings: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Index {
        path: PathBuf,
    },
    Search {
        query: String,
        #[arg(long, value_enum, default_value_t = TierArg::Fast)]
        tier: TierArg,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Purge {
        path: PathBuf,
    },
    Status,
    Watch {
        path: PathBuf,
    },
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum TierArg {
    Instant,
    Fast,
    Deep,
}

impl From<TierArg> for SearchTier {
    fn from(value: TierArg) -> Self {
        match value {
            TierArg::Instant => Self::Instant,
            TierArg::Fast => Self::Fast,
            TierArg::Deep => Self::Deep,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let db_path = cli.db.unwrap_or_else(default_db_path);
    let config = embedding_config(cli.hash_embeddings);
    let mut db = open(&db_path, config).with_context(|| format!("open {}", db_path.display()))?;

    match cli.command {
        Command::Index { path } => {
            let count = db.index_path(&path)?;
            println!("indexed {count} file(s)");
        }
        Command::Search {
            query,
            tier,
            limit,
            json,
        } => {
            let hits = db.search(&SearchOptions {
                query,
                tier: tier.into(),
                limit,
                additional_queries: Vec::new(),
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for hit in hits {
                    println!(
                        "{}:{}-{}  {:.4}",
                        hit.path, hit.line_start, hit.line_end, hit.score
                    );
                    println!("  {}", hit.excerpt);
                }
            }
        }
        Command::Purge { path } => {
            let count = db.purge_path(&path)?;
            println!("purged {count} file(s)");
        }
        Command::Status => {
            let stats = db.stats()?;
            println!("db: {}", stats.db_path.display());
            println!("documents: {}", stats.documents);
            println!("chunks: {}", stats.chunks);
        }
        Command::Watch { path } => watch(path, db)?,
    }
    Ok(())
}

fn embedding_config(hash_embeddings: bool) -> EmbeddingConfig {
    if hash_embeddings {
        EmbeddingConfig {
            backend: EmbeddingBackend::Hash,
            show_download_progress: false,
        }
    } else {
        EmbeddingConfig::default()
    }
}

fn watch(path: PathBuf, mut db: lexa_core::LexaDb) -> anyhow::Result<()> {
    let count = db.index_path(&path)?;
    println!("indexed {count} file(s)");

    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(tx)?;
    watcher.watch(&path, RecursiveMode::Recursive)?;
    println!("watching {}", path.display());

    loop {
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(Ok(_event)) => match db.index_path(&path) {
                Ok(count) => eprintln!("reindexed {count} file(s)"),
                Err(error) => eprintln!("watch reindex failed: {error}"),
            },
            Ok(Err(error)) => eprintln!("watch error: {error}"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => return Err(error.into()),
        }
    }
}
