//! `LexaObsidianDb` — an Obsidian-aware wrapper around `lexa_core::LexaDb`.
//!
//! All retrieval (BM25, dense, hybrid, rerank) and base storage live in
//! `lexa-core`. This crate adds:
//!
//! - Pre-index walk that strips frontmatter and parses Obsidian syntax
//!   (`[[wikilinks]]`, `#tags`, `^block-ids`, `![[embeds]]`) into the
//!   sidecar tables defined in [`schema`].
//! - Note-shaped query API: `search_notes`, `find_backlinks`,
//!   `list_tags`, `get_note`, `get_similar`, `vault_status`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use lexa_core::{
    EmbeddingConfig, IndexStats, LexaDb, LexaError, PreprocessOutput, Preprocessor, SearchOptions,
    SearchTier, Transaction,
};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::frontmatter::{self, Frontmatter};
use crate::tags;
use crate::wikilinks::{self, LinkKind, Wikilink};
use crate::{schema, Result};

/// Wrapper around `LexaDb` with Obsidian-aware indexing and queries.
pub struct LexaObsidianDb {
    inner: LexaDb,
    vault_root: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexReport {
    pub notes_seen: usize,
    pub notes_indexed: usize,
    pub links: usize,
    pub tags: usize,
    pub blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchNotesOptions {
    pub query: String,
    #[serde(default)]
    pub tier: SearchTier,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Path-prefix filters (relative to the vault root).
    #[serde(default)]
    pub folders: Vec<String>,
    #[serde(default)]
    pub additional_queries: Vec<String>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Clone, Serialize)]
pub struct NoteHit {
    pub path: String,
    pub title: String,
    pub score: f32,
    pub excerpt: String,
    pub heading: Option<String>,
    pub line_start: i64,
    pub line_end: i64,
    pub tags: Vec<String>,
    pub breakdown: lexa_core::TierBreakdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct Backlink {
    pub src_path: String,
    pub src_title: Option<String>,
    pub alias: Option<String>,
    pub header: Option<String>,
    pub block_id: Option<String>,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagCount {
    pub tag: String,
    pub count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkRef {
    pub target_name: String,
    pub target_path: Option<String>,
    pub header: Option<String>,
    pub block_id: Option<String>,
    pub alias: Option<String>,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Note {
    pub path: String,
    pub title: String,
    pub frontmatter: serde_json::Value,
    pub body: String,
    pub tags: Vec<String>,
    pub outgoing: Vec<LinkRef>,
    pub incoming: Vec<Backlink>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VaultStatus {
    pub stats: IndexStats,
    pub vault_root: PathBuf,
    pub note_count: i64,
    pub tag_count: i64,
    pub link_count: i64,
    pub needs_index: bool,
}

impl LexaObsidianDb {
    /// Open or create the SQLite-backed index for `vault_root` at
    /// `db_path`, applying both the lexa-core schema and the Obsidian
    /// sidecar schema.
    pub fn open(
        db_path: impl AsRef<Path>,
        vault_root: impl AsRef<Path>,
        embedding_config: EmbeddingConfig,
    ) -> Result<Self> {
        let inner = LexaDb::open(db_path, embedding_config)?;
        schema::migrate(inner.conn())?;
        Ok(Self {
            inner,
            vault_root: vault_root.as_ref().to_path_buf(),
        })
    }

    pub fn vault_root(&self) -> &Path {
        &self.vault_root
    }

    pub fn inner(&self) -> &LexaDb {
        &self.inner
    }

    /// Walk the vault, strip frontmatter before chunking, embed the
    /// **body**, and populate `note_metadata` / `note_tags` /
    /// `note_links` inside the same transaction as the chunk insert.
    /// `note_blocks` is refreshed in a follow-up pass since block-id
    /// extraction needs the persisted `chunks.id`. Idempotent — the
    /// content-hash skip in lexa-core makes re-runs cheap.
    pub fn index_vault(&mut self) -> Result<IndexReport> {
        let mut report = IndexReport {
            notes_seen: 0,
            notes_indexed: 0,
            links: 0,
            tags: 0,
            blocks: 0,
        };

        let preprocessor = ObsidianPreprocessor;

        let report_links = std::cell::Cell::new(0usize);
        let report_tags = std::cell::Cell::new(0usize);

        let indexed = self.inner.index_path_with_preprocessor::<NoteSidecar>(
            &self.vault_root,
            Some(&preprocessor),
            |tx, doc_id, payload| {
                if !payload.is_obsidian_note {
                    return Ok(());
                }
                write_metadata_tx(tx, doc_id, &payload.title, &payload.frontmatter)?;
                replace_tags_tx(tx, doc_id, &payload.tags)?;
                replace_links_tx(tx, doc_id, &payload.links)?;
                report_tags.set(report_tags.get() + payload.tags.len());
                report_links.set(report_links.get() + payload.links.len());
                Ok(())
            },
        )?;
        report.notes_indexed = indexed;
        report.tags = report_tags.get();
        report.links = report_links.get();

        // Block IDs need chunks.id, which only exists post-commit.
        let docs = self.markdown_documents()?;
        report.notes_seen = docs.len();
        for (doc_id, _abs_path) in &docs {
            report.blocks += self.refresh_blocks(*doc_id)?;
        }

        // Final sweep: resolve any wiki-links whose target wasn't yet
        // indexed when the source note's batch flushed.
        self.resolve_pending_links()?;

        Ok(report)
    }

    /// Hybrid retrieval restricted to indexed Obsidian notes.
    pub fn search_notes(&self, opts: &SearchNotesOptions) -> Result<Vec<NoteHit>> {
        let hits = self.inner.search(&SearchOptions {
            query: opts.query.clone(),
            tier: opts.tier,
            limit: opts.limit.saturating_mul(2).max(opts.limit),
            additional_queries: opts.additional_queries.clone(),
        })?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            if !self.path_passes_folder_filter(&hit.path, &opts.folders) {
                continue;
            }
            let doc_id = match self.lookup_doc_id(&hit.path)? {
                Some(id) => id,
                None => continue,
            };
            let tags = self.tags_for_doc(doc_id)?;
            if !opts.tags.is_empty() {
                let note_tags: std::collections::HashSet<&String> = tags.iter().collect();
                if !opts.tags.iter().any(|t| note_tags.contains(t)) {
                    continue;
                }
            }
            let title = self
                .title_for_doc(doc_id)?
                .unwrap_or_else(|| file_stem_of(&hit.path));
            out.push(NoteHit {
                path: hit.path.clone(),
                title,
                score: hit.score,
                excerpt: hit.excerpt.clone(),
                heading: hit.heading.clone(),
                line_start: hit.line_start,
                line_end: hit.line_end,
                tags,
                breakdown: hit.breakdown.clone(),
            });
            if out.len() >= opts.limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn find_backlinks(&self, note: &str) -> Result<Vec<Backlink>> {
        let conn = self.inner.conn();
        let resolved = self.resolve_note_argument(note)?;

        let mut stmt = conn.prepare(
            "SELECT
                d.path,
                m.title,
                nl.alias,
                nl.header,
                nl.block_id,
                nl.kind
             FROM note_links nl
             JOIN documents d ON d.id = nl.src_doc_id
             LEFT JOIN note_metadata m ON m.doc_id = d.id
             WHERE nl.target_path = ?1 OR LOWER(nl.target_name) = LOWER(?2)
             ORDER BY d.path",
        )?;

        let rows = stmt.query_map(
            params![resolved.path.as_deref(), resolved.name.as_str()],
            |row| {
                Ok(Backlink {
                    src_path: row.get(0)?,
                    src_title: row.get::<_, Option<String>>(1)?,
                    alias: row.get::<_, Option<String>>(2)?,
                    header: row.get::<_, Option<String>>(3)?,
                    block_id: row.get::<_, Option<String>>(4)?,
                    kind: row.get(5)?,
                })
            },
        )?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(LexaError::from)
    }

    pub fn list_tags(&self, prefix: Option<&str>, limit: usize) -> Result<Vec<TagCount>> {
        let conn = self.inner.conn();
        let limit = limit.max(1) as i64;
        let rows: Vec<TagCount> = if let Some(prefix) = prefix {
            let pattern = format!("{}%", prefix.to_ascii_lowercase());
            let mut stmt = conn.prepare(
                "SELECT tag, COUNT(*) FROM note_tags
                 WHERE tag LIKE ?1
                 GROUP BY tag ORDER BY COUNT(*) DESC, tag ASC LIMIT ?2",
            )?;
            let rows: Result<Vec<_>> = stmt
                .query_map(params![pattern, limit], |row| {
                    Ok(TagCount {
                        tag: row.get(0)?,
                        count: row.get(1)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(LexaError::from);
            rows?
        } else {
            let mut stmt = conn.prepare(
                "SELECT tag, COUNT(*) FROM note_tags
                 GROUP BY tag ORDER BY COUNT(*) DESC, tag ASC LIMIT ?1",
            )?;
            let rows: Result<Vec<_>> = stmt
                .query_map(params![limit], |row| {
                    Ok(TagCount {
                        tag: row.get(0)?,
                        count: row.get(1)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(LexaError::from);
            rows?
        };
        Ok(rows)
    }

    pub fn get_note(&self, note: &str, block: Option<&str>) -> Result<Note> {
        let resolved = self.resolve_note_argument(note)?;
        let doc_path = resolved
            .path
            .clone()
            .ok_or_else(|| LexaError::InvalidPath(note.to_string()))?;
        let bytes = fs::read(&doc_path)?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let stem = Path::new(&doc_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let (fm, body_str, _) = frontmatter::parse(&text);
        let title = frontmatter::resolve_title(&fm, body_str, stem);
        let body = body_str.to_string();

        let conn = self.inner.conn();
        let doc_id = self
            .lookup_doc_id(&doc_path)?
            .ok_or_else(|| LexaError::InvalidPath(format!("note not indexed: {doc_path}")))?;
        let tags = self.tags_for_doc(doc_id)?;

        let mut outgoing_stmt = conn.prepare(
            "SELECT target_name, target_path, header, block_id, alias, kind
             FROM note_links WHERE src_doc_id = ?1",
        )?;
        let outgoing = outgoing_stmt
            .query_map(params![doc_id], |row| {
                Ok(LinkRef {
                    target_name: row.get(0)?,
                    target_path: row.get::<_, Option<String>>(1)?,
                    header: row.get::<_, Option<String>>(2)?,
                    block_id: row.get::<_, Option<String>>(3)?,
                    alias: row.get::<_, Option<String>>(4)?,
                    kind: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let incoming = self.find_backlinks(&doc_path)?;

        let final_body = if let Some(block_id) = block {
            self.body_for_block(doc_id, &body, block_id)?
                .unwrap_or(body)
        } else {
            body
        };

        Ok(Note {
            path: doc_path,
            title,
            frontmatter: frontmatter_to_json(&fm),
            body: final_body,
            tags,
            outgoing,
            incoming,
        })
    }

    pub fn get_similar(&self, note: &str, limit: usize) -> Result<Vec<NoteHit>> {
        let resolved = self.resolve_note_argument(note)?;
        let doc_path = resolved
            .path
            .ok_or_else(|| LexaError::InvalidPath(note.to_string()))?;
        let bytes = fs::read(&doc_path)?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let (_, body, _) = frontmatter::parse(&text);
        // Take a representative slice; keep small to fit the embedder.
        let snippet: String = body.chars().take(2_000).collect();
        let opts = SearchNotesOptions {
            query: snippet,
            tier: SearchTier::Fast,
            limit: limit.saturating_mul(2).max(limit),
            tags: Vec::new(),
            folders: Vec::new(),
            additional_queries: Vec::new(),
        };
        let hits = self.search_notes(&opts)?;
        Ok(hits
            .into_iter()
            .filter(|h| h.path != doc_path)
            .take(limit)
            .collect())
    }

    pub fn vault_status(&self) -> Result<VaultStatus> {
        let stats = self.inner.stats()?;
        let conn = self.inner.conn();
        let note_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE LOWER(path) LIKE '%.md'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let tag_count: i64 = conn
            .query_row("SELECT COUNT(DISTINCT tag) FROM note_tags", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        let link_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM note_links", [], |row| row.get(0))
            .unwrap_or(0);
        let needs_index = note_count == 0;
        Ok(VaultStatus {
            stats,
            vault_root: self.vault_root.clone(),
            note_count,
            tag_count,
            link_count,
            needs_index,
        })
    }

    pub fn purge_vault(&mut self) -> Result<usize> {
        // CASCADE on documents.id removes sidecar rows automatically.
        self.inner.purge_path(self.vault_root.clone())
    }

    // -------------------- internals --------------------

    fn markdown_documents(&self) -> Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self
            .inner
            .conn()
            .prepare("SELECT id, path FROM documents WHERE LOWER(path) LIKE '%.md' ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            Ok((id, PathBuf::from(path)))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(LexaError::from)
    }

    // metadata / tags / links writes happen inside the lexa-core
    // transaction; see `write_metadata_tx`, `replace_tags_tx`,
    // `replace_links_tx` below.

    fn refresh_blocks(&self, doc_id: i64) -> Result<usize> {
        let conn = self.inner.conn();
        conn.execute("DELETE FROM note_blocks WHERE doc_id = ?1", params![doc_id])?;
        let mut stmt = conn.prepare("SELECT id, text FROM chunks WHERE doc_id = ?1")?;
        let rows = stmt
            .query_map(params![doc_id], |row| {
                let id: i64 = row.get(0)?;
                let text: String = row.get(1)?;
                Ok((id, text))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        let mut inserted = 0usize;
        for (chunk_id, text) in rows {
            if let Some(block_id) = trailing_block_id(&text) {
                conn.execute(
                    "INSERT OR IGNORE INTO note_blocks(chunk_id, doc_id, block_id)
                     VALUES(?1, ?2, ?3)",
                    params![chunk_id, doc_id, block_id],
                )?;
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    fn resolve_pending_links(&self) -> Result<()> {
        let conn = self.inner.conn();
        let mut stmt = conn.prepare("SELECT path FROM documents WHERE LOWER(path) LIKE '%.md'")?;
        let mut by_stem: HashMap<String, String> = HashMap::new();
        for row in stmt.query_map([], |row| row.get::<_, String>(0))? {
            let path = row?;
            let stem = file_stem_of(&path).to_ascii_lowercase();
            by_stem.entry(stem).or_insert(path);
        }
        drop(stmt);

        let mut update_stmt = conn.prepare(
            "UPDATE note_links SET target_path = ?1
             WHERE LOWER(target_name) = ?2 AND target_path IS NULL",
        )?;
        for (stem, path) in &by_stem {
            update_stmt.execute(params![path, stem])?;
        }
        Ok(())
    }

    fn lookup_doc_id(&self, path: &str) -> Result<Option<i64>> {
        let row: Option<i64> = self
            .inner
            .conn()
            .query_row(
                "SELECT id FROM documents WHERE path = ?1",
                params![path],
                |row| row.get(0),
            )
            .optional()?;
        Ok(row)
    }

    fn title_for_doc(&self, doc_id: i64) -> Result<Option<String>> {
        let row: Option<String> = self
            .inner
            .conn()
            .query_row(
                "SELECT title FROM note_metadata WHERE doc_id = ?1",
                params![doc_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(row)
    }

    fn tags_for_doc(&self, doc_id: i64) -> Result<Vec<String>> {
        let mut stmt = self
            .inner
            .conn()
            .prepare("SELECT tag FROM note_tags WHERE doc_id = ?1 ORDER BY tag")?;
        let rows = stmt.query_map(params![doc_id], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn body_for_block(
        &self,
        doc_id: i64,
        full_body: &str,
        block_id: &str,
    ) -> Result<Option<String>> {
        let key = block_id.trim_start_matches('^');
        let mut stmt = self.inner.conn().prepare(
            "SELECT c.text FROM chunks c
             JOIN note_blocks b ON b.chunk_id = c.id
             WHERE b.doc_id = ?1 AND b.block_id = ?2",
        )?;
        let row: Option<String> = stmt
            .query_row(params![doc_id, key], |row| row.get(0))
            .optional()?;
        // If the block isn't in note_blocks (e.g. inline within a chunk),
        // try a substring match against the body.
        if row.is_some() {
            return Ok(row);
        }
        let needle = format!("^{}", key);
        if let Some(idx) = full_body.find(&needle) {
            // Return the paragraph containing the block id.
            let start = full_body[..idx].rfind("\n\n").map(|p| p + 2).unwrap_or(0);
            let end = full_body[idx..]
                .find("\n\n")
                .map(|p| idx + p)
                .unwrap_or(full_body.len());
            return Ok(Some(full_body[start..end].to_string()));
        }
        Ok(None)
    }

    fn resolve_note_argument(&self, note: &str) -> Result<ResolvedNote> {
        // Treat as a path if it exists as a file.
        let candidate = if Path::new(note).is_absolute() {
            PathBuf::from(note)
        } else {
            self.vault_root.join(note)
        };
        if candidate.exists() {
            let canonical = fs::canonicalize(&candidate)?;
            let path = canonical.to_string_lossy().into_owned();
            let name = canonical
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            return Ok(ResolvedNote {
                path: Some(path),
                name,
            });
        }
        // Fall back to the raw stem so backlink lookups can use
        // `LOWER(target_name) = LOWER(?)` even if the note itself isn't
        // indexed (yet).
        let stem = Path::new(note)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(note)
            .to_string();
        let mut stmt = self.inner.conn().prepare(
            "SELECT path FROM documents WHERE LOWER(path) LIKE '%' || LOWER(?1) || '.md'",
        )?;
        let path: Option<String> = stmt.query_row(params![stem], |row| row.get(0)).optional()?;
        Ok(ResolvedNote { path, name: stem })
    }

    fn path_passes_folder_filter(&self, path: &str, folders: &[String]) -> bool {
        if folders.is_empty() {
            return true;
        }
        let path_str = match Path::new(path).strip_prefix(&self.vault_root) {
            Ok(rel) => rel.to_string_lossy().into_owned(),
            Err(_) => path.to_string(),
        };
        folders
            .iter()
            .any(|folder| path_str.starts_with(folder.as_str()))
    }
}

struct ResolvedNote {
    path: Option<String>,
    name: String,
}

fn file_stem_of(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn trailing_block_id(text: &str) -> Option<String> {
    let last = text.lines().rev().find(|l| !l.trim().is_empty())?;
    let trimmed = last.trim();
    let rest = trimmed
        .strip_suffix(|c: char| !c.is_whitespace())
        .map(|_| trimmed)?;
    let _ = rest; // avoid unused warning
    let stripped = trimmed.split_whitespace().last()?;
    let id = stripped.strip_prefix('^')?;
    if id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && !id.is_empty()
    {
        Some(id.to_string())
    } else {
        None
    }
}

fn frontmatter_to_json(fm: &Frontmatter) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    if let Some(title) = &fm.title {
        map.insert("title".into(), serde_json::Value::String(title.clone()));
    }
    if !fm.aliases.is_empty() {
        map.insert(
            "aliases".into(),
            serde_json::Value::Array(
                fm.aliases
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    if !fm.tags.is_empty() {
        map.insert(
            "tags".into(),
            serde_json::Value::Array(
                fm.tags
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    for (k, v) in &fm.raw {
        map.insert(k.clone(), serde_yaml_to_json(v));
    }
    serde_json::Value::Object(map)
}

fn serde_yaml_to_json(value: &serde_yaml::Value) -> serde_json::Value {
    match value {
        serde_yaml::Value::Null => serde_json::Value::Null,
        serde_yaml::Value::Bool(b) => serde_json::Value::Bool(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                serde_json::Value::Number(i.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
        serde_yaml::Value::String(s) => serde_json::Value::String(s.clone()),
        serde_yaml::Value::Sequence(seq) => {
            serde_json::Value::Array(seq.iter().map(serde_yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(m) => {
            let mut out = serde_json::Map::new();
            for (k, v) in m {
                let key = match k {
                    serde_yaml::Value::String(s) => s.clone(),
                    other => serde_yaml::to_string(other)
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                };
                out.insert(key, serde_yaml_to_json(v));
            }
            serde_json::Value::Object(out)
        }
        serde_yaml::Value::Tagged(tagged) => serde_yaml_to_json(&tagged.value),
    }
}

// `LinkKind::as_str` gives us the SQL-stored discriminator. The reverse
// mapping is exposed via `std::str::FromStr` so callers (and tests) can
// recover a `LinkKind` from a database value without a custom helper.
impl std::str::FromStr for LinkKind {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "embed" => LinkKind::Embed,
            _ => LinkKind::Link,
        })
    }
}

/// Per-note sidecar payload threaded from preprocessor → commit hook.
#[derive(Default)]
struct NoteSidecar {
    is_obsidian_note: bool,
    title: String,
    frontmatter: Frontmatter,
    tags: Vec<String>,
    links: Vec<Wikilink>,
}

/// Strips frontmatter from `.md` files so it doesn't leak into the
/// embedding, then captures the parsed metadata for the sidecar tables.
struct ObsidianPreprocessor;

impl Preprocessor for ObsidianPreprocessor {
    type Payload = NoteSidecar;

    fn preprocess(
        &self,
        path: &Path,
        bytes: &[u8],
    ) -> Result<Option<PreprocessOutput<Self::Payload>>> {
        let is_md = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("md"))
            .unwrap_or(false);
        if !is_md {
            return Ok(Some(PreprocessOutput {
                text: String::from_utf8_lossy(bytes).replace("\r\n", "\n"),
                payload: NoteSidecar::default(),
            }));
        }
        let text = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let (fm, body, _offset) = frontmatter::parse(&text);
        let title = frontmatter::resolve_title(&fm, body, stem);
        let extracted_tags = tags::extract(body, &fm);
        let extracted_links = wikilinks::extract(body);
        Ok(Some(PreprocessOutput {
            text: body.to_string(),
            payload: NoteSidecar {
                is_obsidian_note: true,
                title,
                frontmatter: fm,
                tags: extracted_tags,
                links: extracted_links,
            },
        }))
    }
}

fn write_metadata_tx(
    tx: &Transaction<'_>,
    doc_id: i64,
    title: &str,
    fm: &Frontmatter,
) -> lexa_core::Result<()> {
    let aliases_json =
        serde_json::to_string(&fm.aliases).map_err(|err| LexaError::Embedding(err.to_string()))?;
    let raw_yaml = serde_yaml::Value::Mapping(
        fm.raw
            .iter()
            .map(|(k, v)| (serde_yaml::Value::String(k.clone()), v.clone()))
            .collect(),
    );
    let raw_json = serde_json::to_string(&serde_yaml_to_json(&raw_yaml))
        .map_err(|err| LexaError::Embedding(err.to_string()))?;
    tx.execute(
        "INSERT INTO note_metadata(doc_id, title, aliases_json, raw_json)
         VALUES(?1, ?2, ?3, ?4)
         ON CONFLICT(doc_id) DO UPDATE SET
            title = excluded.title,
            aliases_json = excluded.aliases_json,
            raw_json = excluded.raw_json",
        params![doc_id, title, aliases_json, raw_json],
    )?;
    Ok(())
}

fn replace_tags_tx(tx: &Transaction<'_>, doc_id: i64, tags: &[String]) -> lexa_core::Result<()> {
    tx.execute("DELETE FROM note_tags WHERE doc_id = ?1", params![doc_id])?;
    for tag in tags {
        tx.execute(
            "INSERT OR IGNORE INTO note_tags(doc_id, tag) VALUES(?1, ?2)",
            params![doc_id, tag],
        )?;
    }
    Ok(())
}

fn replace_links_tx(
    tx: &Transaction<'_>,
    doc_id: i64,
    links: &[Wikilink],
) -> lexa_core::Result<()> {
    tx.execute(
        "DELETE FROM note_links WHERE src_doc_id = ?1",
        params![doc_id],
    )?;
    for link in links {
        tx.execute(
            "INSERT INTO note_links
                (src_doc_id, target_name, target_path, header, block_id, alias, kind)
             VALUES(?1, ?2, NULL, ?3, ?4, ?5, ?6)",
            params![
                doc_id,
                link.target_name,
                link.header,
                link.block_id,
                link.alias,
                link.kind.as_str(),
            ],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_block_id_extracts_basic() {
        assert_eq!(trailing_block_id("paragraph ^abc-1"), Some("abc-1".into()));
        assert_eq!(trailing_block_id("no marker here"), None);
    }
}
