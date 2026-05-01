use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_char, c_int};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, Transaction};
use walkdir::WalkDir;

use crate::chunk::{chunk_text_for_path, supported_kind};
use crate::embed::{
    matryoshka_truncate, vector_blob, Embedder, EmbeddingConfig, Reranker, EMBEDDING_DIMS,
    PREVIEW_DIMS,
};
use crate::search::{search_impl, SearchOptions};
use crate::types::{Document, IndexStats, LexaError, SearchHit};
use crate::Result;

static SQLITE_VEC: Once = Once::new();
const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;

pub struct LexaDb {
    path: PathBuf,
    conn: Connection,
    embedding_config: EmbeddingConfig,
    embedder: OnceLock<Mutex<Embedder>>,
    reranker: OnceLock<Mutex<Reranker>>,
}

pub fn default_db_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lexa")
        .join("index.sqlite")
}

pub fn open(path: impl AsRef<Path>, embedding_config: EmbeddingConfig) -> Result<LexaDb> {
    LexaDb::open(path, embedding_config)
}

impl LexaDb {
    pub fn open(path: impl AsRef<Path>, embedding_config: EmbeddingConfig) -> Result<Self> {
        register_sqlite_vec();
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        apply_pragmas(&conn)?;
        migrate(&conn)?;
        Ok(Self {
            path,
            conn,
            embedding_config,
            embedder: OnceLock::new(),
            reranker: OnceLock::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn embedder(&self) -> Result<&Mutex<Embedder>> {
        if let Some(cached) = self.embedder.get() {
            return Ok(cached);
        }
        let embedder = Embedder::new(&self.embedding_config)?;
        Ok(self.embedder.get_or_init(|| Mutex::new(embedder)))
    }

    pub fn reranker(&self) -> Result<&Mutex<Reranker>> {
        if let Some(cached) = self.reranker.get() {
            return Ok(cached);
        }
        let reranker = Reranker::new(&self.embedding_config)?;
        Ok(self.reranker.get_or_init(|| Mutex::new(reranker)))
    }

    pub fn index_path(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        const BATCH: usize = 64;
        let files = collect_files(path.as_ref())?;
        let mut prepared: Vec<PreparedDoc> = Vec::new();
        let mut pending_texts: Vec<String> = Vec::new();
        let mut indexed = 0;

        for file in files {
            let Some(doc) = prepare_document(&file)? else {
                continue;
            };
            if self.is_unchanged(&doc)? {
                continue;
            }
            for chunk in &doc.chunks {
                pending_texts.push(match &chunk.context {
                    Some(context) => format!("{context}\n{}", chunk.text),
                    None => chunk.text.clone(),
                });
            }
            prepared.push(doc);

            if prepared.len() >= BATCH {
                indexed += self.flush_batch(&mut prepared, &mut pending_texts)?;
            }
        }
        if !prepared.is_empty() {
            indexed += self.flush_batch(&mut prepared, &mut pending_texts)?;
        }
        Ok(indexed)
    }

    fn is_unchanged(&self, doc: &PreparedDoc) -> Result<bool> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT content_hash FROM documents WHERE path = ?1",
                params![doc.path],
                |row| row.get(0),
            )
            .optional()?;
        Ok(matches!(row, Some(hash) if hash == doc.content_hash))
    }

    fn flush_batch(
        &mut self,
        prepared: &mut Vec<PreparedDoc>,
        pending_texts: &mut Vec<String>,
    ) -> Result<usize> {
        if prepared.is_empty() {
            return Ok(0);
        }
        let embeddings = {
            let lock = self.embedder()?;
            let mut guard = lock
                .lock()
                .map_err(|err| LexaError::Embedding(err.to_string()))?;
            guard.embed_documents(pending_texts)?
        };
        pending_texts.clear();

        // Defensive guard: an embedding of unexpected length would silently
        // corrupt the binary-quantized vector table. Fail fast instead.
        for embedding in &embeddings {
            if embedding.len() != EMBEDDING_DIMS {
                return Err(LexaError::Embedding(format!(
                    "expected {EMBEDDING_DIMS} embedding dims, got {}",
                    embedding.len()
                )));
            }
        }

        let tx = self.conn.transaction()?;
        let mut cursor = 0usize;
        let mut indexed = 0;
        for doc in prepared.drain(..) {
            let count = doc.chunks.len();
            let slice = &embeddings[cursor..cursor + count];
            cursor += count;
            insert_document(&tx, &doc, slice)?;
            indexed += 1;
        }
        tx.commit()?;
        Ok(indexed)
    }

    pub fn purge_path(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        let root = canonical(path.as_ref())?;
        let tx = self.conn.transaction()?;
        let docs = matching_docs(&tx, &root)?;
        for doc in &docs {
            delete_document(&tx, doc.id)?;
        }
        tx.commit()?;
        Ok(docs.len())
    }

    pub fn search(&self, options: &SearchOptions) -> Result<Vec<SearchHit>> {
        search_impl(self, options)
    }

    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn list_documents(&self) -> Result<Vec<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, mtime, size, content_hash, indexed_at FROM documents ORDER BY path",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Document {
                id: row.get(0)?,
                path: row.get(1)?,
                mtime: row.get(2)?,
                size: row.get(3)?,
                content_hash: row.get(4)?,
                indexed_at: row.get(5)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn stats(&self) -> Result<IndexStats> {
        let documents = self
            .conn
            .query_row("SELECT count(*) FROM documents", [], |row| row.get(0))?;
        let chunks = self
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |row| row.get(0))?;
        Ok(IndexStats {
            db_path: self.path.clone(),
            documents,
            chunks,
        })
    }
}

struct PreparedDoc {
    path: String,
    mtime: i64,
    size: i64,
    content_hash: String,
    indexed_at: i64,
    chunks: Vec<crate::chunk::RawChunk>,
}

fn prepare_document(path: &Path) -> Result<Option<PreparedDoc>> {
    let Some(kind) = supported_kind(path) else {
        return Ok(None);
    };
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let text = if kind == "pdf" {
        pdf_extract::extract_text(path).map_err(|error| LexaError::Pdf(error.to_string()))?
    } else {
        if bytes.iter().take(4096).any(|byte| *byte == 0) {
            return Ok(None);
        }
        String::from_utf8_lossy(&bytes).replace("\r\n", "\n")
    };
    let raw_chunks = chunk_text_for_path(&text, kind, Some(path));
    if raw_chunks.is_empty() {
        return Ok(None);
    }
    Ok(Some(PreparedDoc {
        path: canonical(path)?,
        mtime: metadata
            .modified()
            .ok()
            .and_then(epoch_secs)
            .unwrap_or_default() as i64,
        size: metadata.len() as i64,
        content_hash: stable_hash_hex(&bytes),
        indexed_at: epoch_secs(SystemTime::now()).unwrap_or_default() as i64,
        chunks: raw_chunks,
    }))
}

fn insert_document(tx: &Transaction<'_>, doc: &PreparedDoc, embeddings: &[Vec<f32>]) -> Result<()> {
    if let Some(existing_id) = tx
        .query_row(
            "SELECT id FROM documents WHERE path = ?1",
            params![doc.path],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
    {
        delete_document(tx, existing_id)?;
    }
    tx.execute(
        "INSERT INTO documents(path, mtime, size, content_hash, indexed_at) VALUES(?1, ?2, ?3, ?4, ?5)",
        params![doc.path, doc.mtime, doc.size, doc.content_hash, doc.indexed_at],
    )?;
    let doc_id = tx.last_insert_rowid();

    for (idx, (chunk, embedding)) in doc.chunks.iter().zip(embeddings.iter()).enumerate() {
        tx.execute(
            "INSERT INTO chunks(doc_id, ord, byte_start, byte_end, line_start, line_end, kind, text, context)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                doc_id,
                idx as i64,
                chunk.byte_start as i64,
                chunk.byte_end as i64,
                chunk.line_start as i64,
                chunk.line_end as i64,
                chunk.kind,
                chunk.text,
                chunk.context
            ],
        )?;
        let chunk_id = tx.last_insert_rowid();
        let full_blob = vector_blob(embedding);
        let preview_blob = vector_blob(&matryoshka_truncate(embedding, PREVIEW_DIMS));
        tx.execute(
            "INSERT INTO chunks_fts(rowid, text, context) VALUES(?1, ?2, ?3)",
            params![chunk_id, chunk.text, chunk.context.as_deref().unwrap_or("")],
        )?;
        tx.execute(
            "INSERT INTO vectors_bin(rowid, embedding) VALUES(?1, vec_quantize_binary(?2))",
            params![chunk_id, full_blob],
        )?;
        tx.execute(
            "INSERT INTO vectors_bin_preview(rowid, embedding) VALUES(?1, vec_quantize_binary(?2))",
            params![chunk_id, preview_blob],
        )?;
    }
    Ok(())
}

fn register_sqlite_vec() {
    SQLITE_VEC.call_once(|| unsafe {
        type ExtensionEntry = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *const c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> c_int;
        let init = std::mem::transmute::<*const (), ExtensionEntry>(
            sqlite_vec::sqlite3_vec_init as *const (),
        );
        rusqlite::ffi::sqlite3_auto_extension(Some(init));
    });
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "mmap_size", 268_435_456i64)?;
    Ok(())
}

/// Schema migration. The `vec0` virtual table fixes its dim at CREATE time,
/// so `EMBEDDING_DIMS` is interpolated into the DDL. Changing the dim in
/// code requires re-indexing into a fresh DB.
fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            mtime INTEGER NOT NULL,
            size INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ord INTEGER NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end INTEGER NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            context TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            context,
            tokenize='porter unicode61'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vectors_bin USING vec0(embedding bit[{EMBEDDING_DIMS}]);
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors_bin_preview USING vec0(embedding bit[{PREVIEW_DIMS}]);
        "
    ))?;
    Ok(())
}

fn delete_document(tx: &Transaction<'_>, doc_id: i64) -> Result<()> {
    let mut stmt = tx.prepare("SELECT id FROM chunks WHERE doc_id = ?1")?;
    let ids = stmt
        .query_map(params![doc_id], |row| row.get::<_, i64>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);
    for id in ids {
        tx.execute("DELETE FROM chunks_fts WHERE rowid = ?1", params![id])?;
        tx.execute("DELETE FROM vectors_bin WHERE rowid = ?1", params![id])?;
        tx.execute(
            "DELETE FROM vectors_bin_preview WHERE rowid = ?1",
            params![id],
        )?;
    }
    tx.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])?;
    Ok(())
}

fn matching_docs(tx: &Transaction<'_>, root: &str) -> Result<Vec<Document>> {
    let pattern = format!("{root}/%");
    let mut stmt = tx.prepare(
        "SELECT id, path, mtime, size, content_hash, indexed_at
         FROM documents WHERE path = ?1 OR path LIKE ?2",
    )?;
    let rows = stmt.query_map(params![root, pattern], |row| {
        Ok(Document {
            id: row.get(0)?,
            path: row.get(1)?,
            mtime: row.get(2)?,
            size: row.get(3)?,
            content_hash: row.get(4)?,
            indexed_at: row.get(5)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !metadata.is_dir() {
        return Ok(Vec::new());
    }
    let files = WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| !skip_name(entry.file_name().to_string_lossy().as_ref()))
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .collect();
    Ok(files)
}

fn skip_name(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | "node_modules" | ".next" | "dist" | "build" | ".venv"
    )
}

fn canonical(path: &Path) -> Result<String> {
    fs::canonicalize(path)
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(Into::into)
}

fn epoch_secs(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn stable_hash_hex(bytes: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingBackend, SearchTier};

    fn config() -> EmbeddingConfig {
        EmbeddingConfig {
            backend: EmbeddingBackend::Hash,
            show_download_progress: false,
        }
    }

    #[test]
    fn migrations_create_expected_tables() {
        let dir = tempfile::tempdir().unwrap();
        let db = LexaDb::open(dir.path().join("index.sqlite"), config()).unwrap();
        let stats = db.stats().unwrap();
        assert_eq!(stats.documents, 0);
        assert_eq!(stats.chunks, 0);
    }

    #[test]
    fn reindex_replaces_stale_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("repo");
        fs::create_dir_all(&source).unwrap();
        let file = source.join("README.md");
        fs::write(&file, "# Lexa\n\nold search text").unwrap();
        let mut db = LexaDb::open(dir.path().join("index.sqlite"), config()).unwrap();
        assert_eq!(db.index_path(&source).unwrap(), 1);
        fs::write(&file, "# Lexa\n\nconfig validation function").unwrap();
        assert_eq!(db.index_path(&source).unwrap(), 1);
        let stats = db.stats().unwrap();
        assert_eq!(stats.documents, 1);
        assert!(stats.chunks >= 1);
        let hits = db
            .search(&SearchOptions {
                query: "config validation function".to_string(),
                tier: SearchTier::Fast,
                limit: 3,
                additional_queries: Vec::new(),
            })
            .unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].excerpt.contains("config validation"));
    }
}
