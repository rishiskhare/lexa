//! Sidecar schema for the Obsidian adapter.
//!
//! These tables live in the same SQLite file as `lexa-core`'s tables
//! and ride on the same `PRAGMA foreign_keys = ON` so deletes from
//! `documents` cascade through every sidecar.

use rusqlite::Connection;

use crate::Result;

/// Idempotent migration. Safe to run on every open.
pub fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS note_metadata (
            doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            title TEXT,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            raw_json TEXT NOT NULL DEFAULT '{}'
        );

        -- We store every wiki-link occurrence verbatim. Duplicates within
        -- a single source note are rare in practice; the indexer
        -- `DELETE FROM note_links WHERE src_doc_id = ?` before re-insert,
        -- so cumulative duplication can't happen. A surrogate PK keeps
        -- the schema portable (SQLite forbids expressions in composite PKs).
        CREATE TABLE IF NOT EXISTS note_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            target_name TEXT NOT NULL,
            target_path TEXT,
            header TEXT,
            block_id TEXT,
            alias TEXT,
            kind TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_note_links_src ON note_links(src_doc_id);
        CREATE INDEX IF NOT EXISTS idx_note_links_target_path ON note_links(target_path);
        CREATE INDEX IF NOT EXISTS idx_note_links_target_name ON note_links(target_name);

        CREATE TABLE IF NOT EXISTS note_tags (
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            tag TEXT NOT NULL,
            PRIMARY KEY (doc_id, tag)
        );
        CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag);

        CREATE TABLE IF NOT EXISTS note_blocks (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            block_id TEXT NOT NULL,
            UNIQUE (doc_id, block_id)
        );
        CREATE INDEX IF NOT EXISTS idx_note_blocks_doc_block ON note_blocks(doc_id, block_id);
        ",
    )?;
    Ok(())
}
