//! `lexa-obsidian` CLI — Obsidian-aware front-end to the Lexa engine.
//!
//! Subcommands:
//!
//! - `setup` — interactive: pick vault, pre-index, write Codex /
//!   Claude Desktop / Claude Code config, drop AGENTS.md in the vault root.
//! - `doctor` — diagnose every common failure mode in one shot.
//! - `index` — full vault re-index with progress.
//! - `status` — `vault_status` JSON.
//! - `tags` — list tag/count pairs (optionally filtered by prefix).
//! - `backlinks` — list backlinks for a note.
//! - `search` — local hybrid search; `--json` for MCP-shape output.
//! - `watch` — `notify`-based file watcher that re-indexes on change.
//! - `models` — `prefetch` to download embedder + reranker ahead of
//!   time so the first MCP call is instant.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use lexa_core::{EmbeddingBackend, EmbeddingConfig};
use lexa_obsidian::{LexaObsidianDb, NoteHit, SearchNotesOptions};
use notify::{RecursiveMode, Watcher};

#[derive(Debug, Parser)]
#[command(
    name = "lexa-obsidian",
    version,
    about = "Local-first hybrid retrieval over an Obsidian vault. Pairs with \
             `lexa-obsidian-mcp` so any MCP client (Codex, Claude Desktop, \
             Cursor, Claude Code) can answer questions from your notes."
)]
struct Cli {
    /// Vault root. Falls back to `LEXA_OBSIDIAN_VAULT`. Required for
    /// every subcommand except `setup`, `doctor`, and `models`, which
    /// can prompt or operate without a vault.
    #[arg(long, env = "LEXA_OBSIDIAN_VAULT", global = true)]
    vault: Option<PathBuf>,
    /// SQLite path. Defaults to `~/.lexa/obsidian-<sha-of-vault>.sqlite`.
    #[arg(long, env = "LEXA_OBSIDIAN_DB", global = true)]
    db: Option<PathBuf>,
    /// Force the deterministic FNV-1a embedding backend (CI / offline only).
    #[arg(long, global = true)]
    hash_embeddings: bool,
    /// Refuse to make any network calls. Sets `HF_HUB_OFFLINE=1` so the
    /// fastembed model loader errors out rather than silently fetching
    /// from Hugging Face. Useful as a hard offline guarantee after
    /// `lexa-obsidian models prefetch` has populated the cache.
    #[arg(long, global = true)]
    offline: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Interactive bootstrap: pick a vault, pre-index, write client
    /// configs (Codex / Claude Desktop / Claude Code), drop AGENTS.md.
    Setup {
        /// Skip the pre-index step (lets the MCP server lazy-index later).
        #[arg(long)]
        no_index: bool,
        /// Skip writing the Codex `~/.codex/config.toml` block.
        #[arg(long)]
        no_codex: bool,
        /// Also write a Claude Desktop config block.
        #[arg(long)]
        claude_desktop: bool,
        /// Also write a Claude Code config block at `~/.claude.json`.
        #[arg(long)]
        claude_code: bool,
        /// Skip dropping AGENTS.md in the vault root.
        #[arg(long)]
        no_agents_md: bool,
        /// Don't prompt; use defaults / flags. Useful for scripts.
        #[arg(long)]
        non_interactive: bool,
    },
    /// Diagnose every common failure mode in one command.
    Doctor,
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
    /// Manage retrieval models on disk.
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Debug, Subcommand)]
enum ModelAction {
    /// Download Nomic v1.5-Q + BGE-reranker-base ahead of time so the
    /// first MCP / search call doesn't pay the ~390 MB download cost.
    Prefetch,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.offline {
        // The fastembed crate honours `HF_HUB_OFFLINE=1` and refuses
        // every network fetch. Set it before any code path that might
        // initialise an embedder.
        env::set_var("HF_HUB_OFFLINE", "1");
    }
    let config = embedding_config(cli.hash_embeddings);

    match cli.command {
        Command::Setup {
            no_index,
            no_codex,
            claude_desktop,
            claude_code,
            no_agents_md,
            non_interactive,
        } => run_setup(SetupArgs {
            vault_arg: cli.vault.clone(),
            db_arg: cli.db.clone(),
            config,
            no_index,
            no_codex,
            claude_desktop,
            claude_code,
            no_agents_md,
            non_interactive,
        }),
        Command::Doctor => run_doctor(cli.vault.as_deref(), cli.db.as_deref()),
        Command::Models { action } => match action {
            ModelAction::Prefetch => run_prefetch(),
        },
        Command::Index => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let mut db = open_db(&db_path, &vault, config)?;
            let started = Instant::now();
            let report = db.index_vault().context("indexing failed")?;
            let elapsed = started.elapsed().as_secs_f32();
            println!(
                "indexed {} note(s); {} tags, {} links, {} blocks in {:.2}s",
                report.notes_indexed, report.tags, report.links, report.blocks, elapsed,
            );
            Ok(())
        }
        Command::Status => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let db = open_db(&db_path, &vault, config)?;
            let status = db.vault_status().context("loading status failed")?;
            println!("{}", serde_json::to_string_pretty(&status)?);
            Ok(())
        }
        Command::Tags { prefix, limit } => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let db = open_db(&db_path, &vault, config)?;
            let tags = db
                .list_tags(prefix.as_deref(), limit)
                .context("listing tags failed")?;
            for tag in tags {
                println!("{:>5}  #{}", tag.count, tag.tag);
            }
            Ok(())
        }
        Command::Backlinks { note } => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let db = open_db(&db_path, &vault, config)?;
            let backlinks = db
                .find_backlinks(&note)
                .with_context(|| format!("looking up backlinks for {note}"))?;
            for bl in backlinks {
                let title = bl.src_title.unwrap_or_else(|| "<untitled>".into());
                let alias = bl.alias.map(|a| format!(" ({a})")).unwrap_or_default();
                println!("{}: {title}{alias}", bl.kind);
                println!("  {}", bl.src_path);
            }
            Ok(())
        }
        Command::Search {
            query,
            tier,
            limit,
            tag,
            folder,
            json,
        } => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let db = open_db(&db_path, &vault, config)?;
            let parsed_tier = tier.parse().with_context(|| {
                format!("invalid tier '{tier}'; expected one of instant, dense, fast, deep, auto")
            })?;
            let opts = SearchNotesOptions {
                query,
                tier: parsed_tier,
                limit,
                tags: tag,
                folders: folder,
                additional_queries: Vec::new(),
            };
            let hits = db.search_notes(&opts).context("search failed")?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                print_hits(&hits);
            }
            Ok(())
        }
        Command::Watch => {
            let vault = require_vault(&cli.vault)?;
            let db_path = resolve_db_path(cli.db.as_deref(), &vault)?;
            let mut db = open_db(&db_path, &vault, config)?;
            let report = db.index_vault().context("priming index failed")?;
            println!("primed index: {} note(s)", report.notes_indexed);
            let (tx, rx) = channel();
            let mut watcher =
                notify::recommended_watcher(tx).context("creating filesystem watcher failed")?;
            watcher
                .watch(&vault, RecursiveMode::Recursive)
                .with_context(|| format!("watching {} failed", vault.display()))?;
            eprintln!("watching {}", vault.display());
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
}

// =====================================================================
// Helpers
// =====================================================================

fn require_vault(vault: &Option<PathBuf>) -> anyhow::Result<PathBuf> {
    let path = vault.as_ref().ok_or_else(|| {
        anyhow!(
            "vault path is required. Set the LEXA_OBSIDIAN_VAULT env var \
             or pass --vault /path/to/your/Obsidian/vault. Run \
             `lexa-obsidian setup` to configure interactively."
        )
    })?;
    let canonical = fs::canonicalize(path).with_context(|| {
        format!(
            "vault path {} does not exist or is not accessible",
            path.display()
        )
    })?;
    if !canonical.is_dir() {
        return Err(anyhow!(
            "vault path {} is not a directory. Point --vault at the \
             folder that contains your .md files (Obsidian's vault root).",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn resolve_db_path(db: Option<&Path>, vault: &Path) -> anyhow::Result<PathBuf> {
    if let Some(db) = db {
        return Ok(db.to_path_buf());
    }
    let dir = home_dir()?.join(".lexa");
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir.join(format!("obsidian-{}.sqlite", vault_hash(vault))))
}

fn open_db(
    db_path: &Path,
    vault: &Path,
    config: EmbeddingConfig,
) -> anyhow::Result<LexaObsidianDb> {
    LexaObsidianDb::open(db_path, vault, config)
        .with_context(|| format!("opening Lexa DB at {}", db_path.display()))
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

fn home_dir() -> anyhow::Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set; cannot determine ~/.lexa path"))
}

fn vault_hash(path: &Path) -> String {
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

fn read_line(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    Ok(buf.trim().to_string())
}

fn confirm(prompt: &str, default_yes: bool) -> anyhow::Result<bool> {
    let suffix = if default_yes { "[Y/n]" } else { "[y/N]" };
    let answer = read_line(&format!("{prompt} {suffix} "))?;
    Ok(match answer.to_ascii_lowercase().as_str() {
        "" => default_yes,
        "y" | "yes" => true,
        _ => false,
    })
}

// =====================================================================
// `setup` — interactive bootstrap
// =====================================================================

struct SetupArgs {
    vault_arg: Option<PathBuf>,
    db_arg: Option<PathBuf>,
    config: EmbeddingConfig,
    no_index: bool,
    no_codex: bool,
    claude_desktop: bool,
    claude_code: bool,
    no_agents_md: bool,
    non_interactive: bool,
}

fn run_setup(args: SetupArgs) -> anyhow::Result<()> {
    println!("=== lexa-obsidian setup ===");
    println!("Local-first retrieval over your Obsidian vault. Nothing leaves your machine.\n");

    let vault = match args.vault_arg {
        Some(v) => v,
        None if args.non_interactive => {
            return Err(anyhow!(
                "--non-interactive setup requires --vault or LEXA_OBSIDIAN_VAULT"
            ))
        }
        None => {
            let raw = read_line("Vault path (your Obsidian vault root): ")?;
            if raw.is_empty() {
                return Err(anyhow!("vault path is required to continue"));
            }
            PathBuf::from(expand_tilde(&raw))
        }
    };
    let canonical_vault = fs::canonicalize(&vault).with_context(|| {
        format!(
            "vault path {} does not exist. Set --vault to the directory \
             that holds your .md files.",
            vault.display()
        )
    })?;
    if !canonical_vault.is_dir() {
        return Err(anyhow!(
            "vault path {} is not a directory",
            canonical_vault.display()
        ));
    }
    let note_count = count_markdown_files(&canonical_vault);
    println!(
        "✓ vault: {} ({note_count} .md files)",
        canonical_vault.display()
    );

    let db_path = resolve_db_path(args.db_arg.as_deref(), &canonical_vault)?;
    println!("✓ index DB: {}", db_path.display());

    let do_index = if args.no_index {
        false
    } else if args.non_interactive {
        true
    } else {
        let est = estimate_index_seconds(note_count);
        confirm(
            &format!(
                "Pre-index now? Recommended for >1000-note vaults. Estimated ~{est:.0}s on real embeddings."
            ),
            true,
        )?
    };

    if do_index {
        println!("Pre-indexing — this can take several minutes on a large vault…");
        let mut db = open_db(&db_path, &canonical_vault, args.config.clone())?;
        let started = Instant::now();
        let report = db.index_vault().context("pre-indexing the vault failed")?;
        println!(
            "✓ indexed {} note(s); {} tags, {} links, {} blocks in {:.2}s",
            report.notes_indexed,
            report.tags,
            report.links,
            report.blocks,
            started.elapsed().as_secs_f32()
        );
    } else {
        println!("(skipping pre-index — the MCP server will index on first call)");
    }

    let codex = !args.no_codex
        && (args.non_interactive || confirm("Configure Codex CLI (~/.codex/config.toml)?", true)?);
    let claude_desktop = args.claude_desktop
        || (!args.non_interactive
            && confirm(
                "Also configure Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json)?",
                false,
            )?);
    let claude_code = args.claude_code
        || (!args.non_interactive
            && confirm("Also configure Claude Code (~/.claude.json)?", false)?);
    let drop_agents_md = !args.no_agents_md
        && (args.non_interactive
            || confirm(
                "Drop AGENTS.md in the vault root so MCP-aware agents don't need 'Use lexa-obsidian.' before every prompt?",
                true,
            )?);

    let mcp_server_path = which_lexa_obsidian_mcp();
    let mcp_command = mcp_server_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "lexa-obsidian-mcp".to_string());
    let vault_str = canonical_vault.to_string_lossy().into_owned();

    if codex {
        let path = home_dir()?.join(".codex").join("config.toml");
        configure_codex(&path, &mcp_command, &vault_str)?;
        println!("✓ wrote {} (Codex MCP server registration)", path.display());
    }
    if claude_desktop {
        let path = home_dir()?
            .join("Library")
            .join("Application Support")
            .join("Claude")
            .join("claude_desktop_config.json");
        configure_claude_json(&path, "lexa-obsidian", &mcp_command, &vault_str)?;
        println!("✓ wrote {} (Claude Desktop)", path.display());
    }
    if claude_code {
        let path = home_dir()?.join(".claude.json");
        configure_claude_json(&path, "lexa-obsidian", &mcp_command, &vault_str)?;
        println!("✓ wrote {} (Claude Code)", path.display());
    }

    if drop_agents_md {
        let agents_path = canonical_vault.join("AGENTS.md");
        if agents_path.exists() {
            println!(
                "(skipping {} — already exists; merge `templates/AGENTS.md` from the lexa repo manually if you want)",
                agents_path.display()
            );
        } else {
            fs::write(&agents_path, AGENTS_MD_TEMPLATE)
                .with_context(|| format!("writing {}", agents_path.display()))?;
            println!("✓ wrote {}", agents_path.display());
        }
    }

    println!();
    println!("Done. Restart your MCP client and try one of:");
    println!("  > what did I write about <topic>?");
    println!("  > list my top 10 tags");
    println!("  > show me backlinks for <note name>");
    println!();
    println!("If something looks wrong, run `lexa-obsidian doctor`.");
    Ok(())
}

const AGENTS_MD_TEMPLATE: &str = r#"# Working with this Obsidian vault

You have access to the **lexa-obsidian** MCP server, which indexes the
notes in this vault with hybrid retrieval (BM25 + dense + cross-encoder
rerank, all local).

For any question that could be answered from the user's notes —
content lookup, recall, summarisation, backlinks, tags, "find a note
about X" — call the appropriate `lexa-obsidian` tool **before**
answering from prior knowledge.

| User asks                          | Call                |
|------------------------------------|---------------------|
| "what did I write about X?"        | `search_notes`      |
| "show me note Y" / "expand block Z"| `get_note`          |
| "what links to Y?"                 | `find_backlinks`    |
| "find notes similar to Y"          | `get_similar`       |
| "what tags do I use most?"         | `list_tags`         |

Default the `tier` argument of `search_notes` to `auto`. Don't add
`Use lexa-obsidian.` to your responses; the routing should be implicit.
"#;

fn estimate_index_seconds(note_count: usize) -> f32 {
    // ~90 ms / note on real Nomic v1.5-Q (measured 2026-05-01).
    note_count as f32 * 0.09
}

fn count_markdown_files(root: &Path) -> usize {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("md"))
                    .unwrap_or(false)
        })
        .count()
}

fn which_lexa_obsidian_mcp() -> Option<PathBuf> {
    // Prefer the binary next to the running CLI, then walk PATH.
    if let Ok(self_exe) = env::current_exe() {
        if let Some(parent) = self_exe.parent() {
            let candidate = parent.join(if cfg!(windows) {
                "lexa-obsidian-mcp.exe"
            } else {
                "lexa-obsidian-mcp"
            });
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(if cfg!(windows) {
            "lexa-obsidian-mcp.exe"
        } else {
            "lexa-obsidian-mcp"
        });
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn expand_tilde(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home)
                .join(rest)
                .to_string_lossy()
                .into_owned();
        }
    }
    s.to_string()
}

fn configure_codex(path: &Path, command: &str, vault: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut updated = existing.clone();

    let server_block = format!(
        "\n[mcp_servers.lexa-obsidian]\ncommand = \"{cmd}\"\nenv = {{ LEXA_OBSIDIAN_VAULT = \"{vault}\" }}\n",
        cmd = command.replace('"', "\\\""),
        vault = vault.replace('"', "\\\""),
    );
    if !updated.contains("[mcp_servers.lexa-obsidian]") {
        if !updated.ends_with('\n') && !updated.is_empty() {
            updated.push('\n');
        }
        updated.push_str(&server_block);
    }

    let instructions_block = "\n[default_session]\n\
        instructions = \"\"\"\n\
        You have access to the lexa-obsidian MCP server, which indexes the user's Obsidian vault.\n\
        For any question that could be answered from their notes — content lookup, recall, summarisation,\n\
        backlinks, tags, 'what did I write about X' — call the appropriate lexa-obsidian tool BEFORE\n\
        answering from prior knowledge. Default the search tier to 'auto'.\n\
        \"\"\"\n";
    if !updated.contains("[default_session]") {
        if !updated.ends_with('\n') && !updated.is_empty() {
            updated.push('\n');
        }
        updated.push_str(instructions_block);
    }

    if updated != existing {
        fs::write(path, updated).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn configure_claude_json(
    path: &Path,
    server_name: &str,
    command: &str,
    vault: &str,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let existing: serde_json::Value = if path.exists() {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if bytes.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {} as JSON", path.display()))?
        }
    } else {
        serde_json::json!({})
    };
    let mut root = existing;
    let object = root.as_object_mut().ok_or_else(|| {
        anyhow!(
            "{} exists but isn't a JSON object; merge the lexa-obsidian \
             entry by hand or move the file aside",
            path.display()
        )
    })?;
    let servers = object
        .entry("mcpServers".to_string())
        .or_insert_with(|| serde_json::json!({}));
    let servers_obj = servers.as_object_mut().ok_or_else(|| {
        anyhow!(
            "{} has a non-object `mcpServers` entry; cannot merge",
            path.display()
        )
    })?;
    servers_obj.insert(
        server_name.to_string(),
        serde_json::json!({
            "command": command,
            "env": { "LEXA_OBSIDIAN_VAULT": vault },
        }),
    );
    let serialised = serde_json::to_string_pretty(&root)?;
    fs::write(path, serialised).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

// =====================================================================
// `doctor` — diagnose every common failure mode
// =====================================================================

fn run_doctor(vault: Option<&Path>, db_arg: Option<&Path>) -> anyhow::Result<()> {
    println!("=== lexa-obsidian doctor ===");
    let mut ok = true;

    // 1. Check both binaries are findable.
    match env::current_exe() {
        Ok(p) => println!("✓ lexa-obsidian binary at {}", p.display()),
        Err(err) => {
            println!("✗ cannot locate own binary: {err}");
            ok = false;
        }
    }
    match which_lexa_obsidian_mcp() {
        Some(p) => println!("✓ lexa-obsidian-mcp binary at {}", p.display()),
        None => {
            println!(
                "✗ lexa-obsidian-mcp not found. Install with: cargo install \
                 --path crates/lexa-obsidian (or grab a release tarball)."
            );
            ok = false;
        }
    }

    // 2. Check vault if known.
    let canonical_vault = match vault {
        Some(v) => match fs::canonicalize(v) {
            Ok(c) if c.is_dir() => {
                let n = count_markdown_files(&c);
                println!("✓ vault: {} ({n} .md files)", c.display());
                Some(c)
            }
            Ok(c) => {
                println!("✗ vault path {} is not a directory", c.display());
                ok = false;
                None
            }
            Err(err) => {
                println!("✗ vault path {} unreadable: {err}", v.display());
                ok = false;
                None
            }
        },
        None => {
            println!(
                "… vault path not provided. Set LEXA_OBSIDIAN_VAULT or pass --vault to check it."
            );
            None
        }
    };

    // 3. Check DB.
    if let Some(vault) = canonical_vault.as_deref() {
        let db_path = resolve_db_path(db_arg, vault)?;
        if db_path.exists() {
            match LexaObsidianDb::open(&db_path, vault, EmbeddingConfig::default()) {
                Ok(db) => match db.vault_status() {
                    Ok(status) => println!(
                        "✓ index DB at {} ({} notes, {} tags, {} links{})",
                        db_path.display(),
                        status.note_count,
                        status.tag_count,
                        status.link_count,
                        if status.needs_index {
                            ", needs reindex"
                        } else {
                            ""
                        }
                    ),
                    Err(err) => {
                        println!("✗ DB at {} unreadable: {err}", db_path.display());
                        ok = false;
                    }
                },
                Err(err) => {
                    println!("✗ opening DB at {} failed: {err}", db_path.display());
                    ok = false;
                }
            }
        } else {
            println!(
                "… index DB at {} does not exist yet. Run `lexa-obsidian index`.",
                db_path.display()
            );
        }
    }

    // 4. Models cache.
    let cache = fastembed_cache_dir();
    if cache.exists() {
        let nomic = cache.join("models--nomic-ai--nomic-embed-text-v1.5");
        let bge = cache.join("models--Xenova--bge-reranker-base").exists()
            || cache.join("models--BAAI--bge-reranker-base").exists();
        let nomic_present = nomic.exists();
        match (nomic_present, bge) {
            (true, true) => println!("✓ retrieval models cached at {}", cache.display()),
            (true, false) => println!(
                "… Nomic embedder cached, but BGE reranker not yet downloaded. \
                 First `deep`-tier search will pull ~280 MB."
            ),
            (false, true) => println!(
                "… BGE reranker cached, but Nomic embedder not yet downloaded. \
                 First search will pull ~110 MB."
            ),
            (false, false) => println!(
                "… retrieval models not cached at {}. Run `lexa-obsidian \
                 models prefetch` to download (~390 MB total).",
                cache.display()
            ),
        }
    } else {
        println!(
            "… fastembed cache directory not yet created. Run `lexa-obsidian models prefetch` \
             to download retrieval models (~390 MB)."
        );
    }

    // 5. Codex config.
    if let Ok(home) = env::var("HOME") {
        let codex_path = PathBuf::from(&home).join(".codex").join("config.toml");
        match fs::read_to_string(&codex_path) {
            Ok(contents) if contents.contains("[mcp_servers.lexa-obsidian]") => {
                println!("✓ ~/.codex/config.toml has the lexa-obsidian MCP block");
            }
            Ok(_) => {
                println!(
                    "… ~/.codex/config.toml exists but no [mcp_servers.lexa-obsidian] block. Run \
                     `lexa-obsidian setup` to add it."
                );
            }
            Err(_) => {
                println!(
                    "… ~/.codex/config.toml does not exist yet. Run `lexa-obsidian setup` to write it."
                );
            }
        }
    }

    println!();
    if ok {
        println!("All systems go. Try Codex with: 'what did I write about <topic>?'");
        Ok(())
    } else {
        Err(anyhow!(
            "one or more checks failed; address the ✗ items above"
        ))
    }
}

fn fastembed_cache_dir() -> PathBuf {
    if let Ok(custom) = env::var("FASTEMBED_CACHE_PATH") {
        return PathBuf::from(custom);
    }
    let cwd_cache = PathBuf::from(".fastembed_cache");
    if cwd_cache.exists() {
        return cwd_cache;
    }
    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("fastembed");
    }
    cwd_cache
}

// =====================================================================
// `models prefetch`
// =====================================================================

fn run_prefetch() -> anyhow::Result<()> {
    println!(
        "Downloading retrieval models (~390 MB on first run; subsequent runs reuse the cache)…"
    );
    let config = EmbeddingConfig {
        backend: EmbeddingBackend::FastEmbed,
        show_download_progress: true,
    };
    // Open a temporary DB so we can hit the cached embedder + reranker
    // accessors. Using an in-memory path avoids cluttering ~/.lexa.
    let temp_db = env::temp_dir().join(format!("lexa-prefetch-{}.sqlite", std::process::id()));
    let _cleanup = scopeguard::guard((), |_| {
        let _ = fs::remove_file(&temp_db);
        let _ = fs::remove_file(temp_db.with_extension("sqlite-wal"));
        let _ = fs::remove_file(temp_db.with_extension("sqlite-shm"));
    });
    let db = lexa_core::open(&temp_db, config).context("preparing temporary DB")?;
    {
        let lock = db.embedder().context("loading embedder")?;
        drop(
            lock.lock()
                .map_err(|err| anyhow!("embedder lock poisoned: {err}"))?,
        );
    }
    {
        let lock = db.reranker().context("loading reranker")?;
        drop(
            lock.lock()
                .map_err(|err| anyhow!("reranker lock poisoned: {err}"))?,
        );
    }
    println!("✓ models cached. Future MCP / search calls won't pay the download cost.");
    Ok(())
}

// Tiny inline scopeguard to avoid pulling a dependency.
mod scopeguard {
    pub struct Guard<T, F: FnMut(T)> {
        value: Option<T>,
        run: F,
    }
    impl<T, F: FnMut(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let Some(v) = self.value.take() {
                (self.run)(v);
            }
        }
    }
    pub fn guard<T, F: FnMut(T)>(value: T, run: F) -> Guard<T, F> {
        Guard {
            value: Some(value),
            run,
        }
    }
}
