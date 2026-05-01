//! `lexa-obsidian` CLI — Obsidian-aware front-end to the Lexa engine.
//!
//! Commands:
//!
//! - `index`     — full vault re-index with progress.
//! - `status`    — emit `vault_status` JSON.
//! - `tags`      — list tag/count pairs (optionally filtered by prefix).
//! - `backlinks` — list backlinks for a note.
//! - `search`    — local hybrid search; `--json` for MCP-shape output.
//! - `watch`     — `notify`-based file watcher that re-indexes on change.

use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Parser, Subcommand};
use lexa_core::{EmbeddingBackend, EmbeddingConfig};
use lexa_obsidian::{LexaObsidianDb, NoteHit, SearchNotesOptions};
use notify::{RecursiveMode, Watcher};

#[derive(Debug, Parser)]
#[command(
    name = "lexa-obsidian",
    version,
    about = "Local-first hybrid retrieval over an Obsidian vault."
)]
struct Cli {
    /// Vault root. Falls back to `LEXA_OBSIDIAN_VAULT`.
    #[arg(long, env = "LEXA_OBSIDIAN_VAULT")]
    vault: PathBuf,
    /// SQLite path. Defaults to `~/.lexa/<sha-of-vault>.sqlite`.
    #[arg(long, env = "LEXA_OBSIDIAN_DB")]
    db: Option<PathBuf>,
    /// Force the deterministic FNV-1a embedding backend (CI / offline only).
    #[arg(long)]
    hash_embeddings: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Walk the vault and (re)build the index. Idempotent.
    Index,
    /// Emit `vault_status` as JSON.
    Status,
    /// List tags ranked by note count.
    Tags {
        #[arg(long)]
        prefix: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Print backlinks for a note (path or filename stem).
    Backlinks { note: String },
    /// Hybrid search; default tier is `auto`.
    Search {
        query: String,
        #[arg(long, default_value = "auto")]
        tier: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Filter by tag (may be passed multiple times).
        #[arg(long)]
        tag: Vec<String>,
        /// Filter by relative folder prefix (may be passed multiple times).
        #[arg(long)]
        folder: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Re-index on every change event under the vault root.
    Watch,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let db_path = match &cli.db {
        Some(path) => path.clone(),
        None => default_db_path(&cli.vault)?,
    };
    let config = embedding_config(cli.hash_embeddings);

    match cli.command {
        Command::Index => {
            let mut db = open(&db_path, &cli.vault, config)?;
            let started = Instant::now();
            let report = db.index_vault()?;
            let elapsed = started.elapsed().as_secs_f32();
            println!(
                "indexed {} note(s); {} tags, {} links, {} blocks in {:.2}s",
                report.notes_indexed, report.tags, report.links, report.blocks, elapsed,
            );
        }
        Command::Status => {
            let db = open(&db_path, &cli.vault, config)?;
            let status = db.vault_status()?;
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        Command::Tags { prefix, limit } => {
            let db = open(&db_path, &cli.vault, config)?;
            let tags = db.list_tags(prefix.as_deref(), limit)?;
            for tag in tags {
                println!("{:>5}  #{}", tag.count, tag.tag);
            }
        }
        Command::Backlinks { note } => {
            let db = open(&db_path, &cli.vault, config)?;
            let backlinks = db.find_backlinks(&note)?;
            for bl in backlinks {
                let title = bl.src_title.unwrap_or_else(|| "<untitled>".into());
                let alias = bl.alias.map(|a| format!(" ({a})")).unwrap_or_default();
                println!("{}: {}{}", bl.kind, title, alias);
                println!("  {}", bl.src_path);
            }
        }
        Command::Search {
            query,
            tier,
            limit,
            tag,
            folder,
            json,
        } => {
            let db = open(&db_path, &cli.vault, config)?;
            let parsed_tier = tier
                .parse()
                .with_context(|| format!("invalid tier: {tier}"))?;
            let opts = SearchNotesOptions {
                query,
                tier: parsed_tier,
                limit,
                tags: tag,
                folders: folder,
                additional_queries: Vec::new(),
            };
            let hits = db.search_notes(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                print_hits(&hits);
            }
        }
        Command::Watch => {
            let mut db = open(&db_path, &cli.vault, config)?;
            let report = db.index_vault()?;
            println!("primed index: {} note(s)", report.notes_indexed);
            let (tx, rx) = channel();
            let mut watcher = notify::recommended_watcher(tx)?;
            watcher.watch(&cli.vault, RecursiveMode::Recursive)?;
            eprintln!("watching {}", cli.vault.display());
            loop {
                match rx.recv_timeout(Duration::from_secs(3600)) {
                    Ok(Ok(_)) => match db.index_vault() {
                        Ok(rep) => eprintln!(
                            "reindexed: {} note(s), {} tags, {} links, {} blocks",
                            rep.notes_indexed, rep.tags, rep.links, rep.blocks
                        ),
                        Err(err) => eprintln!("watch reindex failed: {err}"),
                    },
                    Ok(Err(err)) => eprintln!("watch error: {err}"),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(err) => return Err(err.into()),
                }
            }
        }
    }
    Ok(())
}

fn open(db_path: &Path, vault: &Path, config: EmbeddingConfig) -> anyhow::Result<LexaObsidianDb> {
    Ok(LexaObsidianDb::open(db_path, vault, config)?)
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

/// Per-vault database path: `~/.lexa/obsidian-<8-char-hash>.sqlite`.
/// Hash the canonicalised vault path so two distinct vaults don't share a DB.
fn default_db_path(vault: &Path) -> anyhow::Result<PathBuf> {
    let canonical = std::fs::canonicalize(vault)
        .with_context(|| format!("vault path does not exist: {}", vault.display()))?;
    let hash = vault_hash(&canonical);
    let dir = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lexa");
    std::fs::create_dir_all(&dir).context("creating ~/.lexa")?;
    Ok(dir.join(format!("obsidian-{hash}.sqlite")))
}

fn vault_hash(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn print_hits(hits: &[NoteHit]) {
    for hit in hits {
        println!(
            "{}:{}-{}  {:.4}  {}",
            hit.path, hit.line_start, hit.line_end, hit.score, hit.title
        );
        if let Some(heading) = &hit.heading {
            println!("  # {heading}");
        }
        let snippet = hit
            .excerpt
            .lines()
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {snippet}");
        if !hit.tags.is_empty() {
            let tags = hit
                .tags
                .iter()
                .map(|t| format!("#{t}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("  {tags}");
        }
    }
}
