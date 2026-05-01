# Lexa Architecture

Lexa is structured as four crates inside one Cargo workspace:

| Crate        | Role                                                                          |
|--------------|-------------------------------------------------------------------------------|
| `lexa-core`  | Storage, chunking, embeddings, retrieval, fusion. Library only.               |
| `lexa-cli`   | The `lexa` binary — `index`, `search`, `purge`, `status`, `watch`.            |
| `lexa-mcp`   | `lexa-mcp` binary — rmcp stdio server exposing search tools to MCP clients.   |
| `lexa-bench` | `lexa-bench` binary — five reproducible benchmark harnesses (A–E).            |

Everything else is a single static binary, no daemon, no Docker, no Python.

## Storage

A single SQLite file (`~/.lexa/index.sqlite` by default) holds the entire
index. Tables:

- `documents` — one row per file (path, mtime, size, content hash,
  indexed_at).
- `chunks` — one row per chunk (doc_id, ord, byte and line offsets, kind,
  text, optional context).
- `chunks_fts` — FTS5 virtual table over `chunks.text` and
  `chunks.context`, `tokenize='porter unicode61'`.
- `vectors_bin` — `vec0(embedding bit[768])` from `sqlite-vec`. Holds the
  full 768-dim Nomic embedding, binary-quantized at insert time via
  `vec_quantize_binary()`.
- `vectors_bin_preview` — `vec0(embedding bit[256])`. Holds the
  Matryoshka-truncated 256-dim prefix of every chunk embedding,
  binary-quantized. Drives the first stage of the two-stage KNN.

`PRAGMA journal_mode = WAL`, `synchronous = NORMAL`, `mmap_size = 256 MB`,
`temp_store = MEMORY`. WAL lets the MCP server share an index with the
CLI cleanly.

## Retrieval

`crates/lexa-core/src/search.rs::search_impl` dispatches on
`SearchTier { Instant, Dense, Fast, Deep, Auto }`.

- **`Instant`** — FTS5 BM25 only. No model load, no vector access.
- **`Dense`** — Two-stage Matryoshka KNN only. Coarse 256-bit Hamming
  KNN over `vectors_bin_preview` returns 8× the final K, then the
  full 768-bit Hamming distance is computed for the survivors via
  `vec_distance_hamming` to produce the top K. One ONNX forward pass per
  query.
- **`Fast`** — BM25 ⊕ Dense, fused with Reciprocal Rank Fusion at
  `k = 60` (the Cormack et al. 2009 constant Exa also uses). The BM25
  SQL hit and the embedder ONNX forward pass run in parallel via
  `std::thread::scope`; their results meet at the binary KNN.
- **`Deep`** — `Fast` plus cross-encoder reranking. Top 15 fused
  candidates are sent to `BAAI/bge-reranker-base`; the cross-encoder
  logit is sigmoid-squashed and blended (`α = 0.7`) with the RRF score
  rather than overriding it. If `SearchOptions::additional_queries` is
  non-empty, those reformulations are run through the same hybrid
  pipeline in parallel and RRF-fused into the candidate set before the
  reranker sees it (this is the Exa Deep `additionalQueries` feature).
- **`Auto`** — The default. Inspects the query and routes:
  - explicit `[deep]` prefix → `Deep`
  - exactly 1 token shaped like an identifier (snake_case, mixed-case
    CamelCase, or `::` / `.` path) → `Instant`
  - 6+ tokens ending with `?` → `Deep`
  - otherwise → `Fast`

## Chunking

`crates/lexa-core/src/chunk.rs` dispatches by file extension:

- **Code** (`.rs`, `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.java`, `.c`,
  `.cpp`, `.h`): tree-sitter parser walks the AST and emits one chunk per
  function / method / class / impl block; leading comments are folded
  into the chunk so the documentation travels with the symbol.
- **Markdown**: split on heading boundaries (`# ` through `###### `).
- **Plain text-like** (`.txt`, `.log`, `.json`, `.toml`, `.yaml`, `.csv`):
  stable line windows.
- **PDF** (`.pdf`): `pdf-extract` produces text, then the plain-text
  chunker runs.

Every chunk records byte and line offsets, so search results carry
`file:line_start-line_end` references suitable for editors and agents.

## Embeddings

`crates/lexa-core/src/embed.rs` ships exactly one production embedder and
one production reranker:

- **Embedder**: `nomic-ai/nomic-embed-text-v1.5` (quantized) via
  fastembed-rs's ONNX backend. 768 dimensions. Apache-2 licensed.
  Matryoshka-trained at canonical prefix dims `{64, 128, 256, 512, 768}`,
  which is what makes the two-stage retrieval safe.
- **Reranker**: `BAAI/bge-reranker-base` cross-encoder via fastembed-rs.

Both are cached on `LexaDb` via `OnceLock<Mutex<…>>`; the ONNX models
load once per process. Queries get the `search_query: ` task prefix and
documents the `search_document: ` task prefix automatically — the
asymmetric pair Nomic was trained with.

A deterministic FNV-1a hash backend is available for tests and CI smoke
runs only (`LEXA_EMBEDDER=hash`).

## Highlights (excerpt extraction)

`search.rs::highlight` produces the snippet shown in `SearchHit::excerpt`.
Algorithm:

1. Tokenize the query with the FTS5 tokenizer (lowercase, drop short
   tokens, drop the curated stopword set).
2. Split the chunk into sentence spans on `[.!?;\n]\s+`.
3. Score each sentence by *unique* query-token overlap.
4. Pick the highest-scoring sentence; expand by ±1 sentence until the
   span reaches ~220 chars; cap at 1.5× the soft target.
5. Fall back to a plain truncation `excerpt` only if no sentence shares
   any token with the query.

This mirrors the **highlights** concept in Exa's contents API: a short,
LLM-friendly span containing the answer rather than the whole chunk.

## MCP

`crates/lexa-mcp/src/main.rs` uses the official Anthropic `rmcp` crate
with `transport-io` (stdio). Tools:

- `search_files(query, tier?, limit?)` → top-K hits.
- `index_path(path)` → idempotent index (content-hash skip).
- `list_indexed_paths()` → registered paths from the `documents` table.
- `purge_path(path)` → removes a path subtree.
- `status()` → DB path + counts.

Logs go to stderr only; stdout is reserved for the JSON-RPC stream. The
server never executes user-supplied commands.

## Why these design choices

See:

- `docs/adr/000-name.md` — name
- `docs/adr/001-sqlite-vec.md` — why SQLite + FTS5 + sqlite-vec
- `docs/adr/002-embeddings.md` — why Nomic v1.5-Q + BGE reranker
- `docs/adr/003-chunking.md` — chunking by file type
- `docs/adr/004-tiers.md` — the five-tier ladder
- `docs/adr/005-mcp.md` — MCP posture
