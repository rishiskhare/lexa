# Lexa — local Exa

> Hybrid retrieval over your local files and code, in a single static Rust
> binary. Lexa applies the architecture of [Exa](https://exa.ai/) — five
> latency-tiered search modes, hybrid BM25 + dense + RRF, two-stage
> Matryoshka KNN, binary-quantized vectors, query-aware highlights, deep
> reranking with optional query expansion, LLM-as-judge evaluation — to
> the corpus already on your disk.

```bash
lexa index ~/repos/myproject
lexa search "where does the rate limiter back off when redis is down"
```

```text
crates/api/src/limiter.rs:48-72   0.7141
  if !backend.is_healthy().await { tracing::warn!("redis down, switching to in-memory backoff");
   return self.fallback.acquire(key).await; }
```

## Highlights

- **Single static binary**, no daemon, no Python, no Docker. SQLite (with
  FTS5 and `sqlite-vec`) is the entire backend.
- **Sub-10 ms `fast` tier** on real Nomic-v1.5 embeddings (M-series
  warm-state, 2 000 docs, 500 iterations). 38× faster than the published
  Exa Fast latency budget.
- **Five search tiers** — `instant`, `dense`, `fast`, `deep`, `auto` —
  mirroring Exa's tiered API.
- **Two-stage Matryoshka KNN** (256-bit preview → 768-bit re-score) the
  same way Exa runs prefix-256 over their 4096-dim embeddings.
- **Deep tier with query expansion** (`additional_queries`) and a sigmoid-
  blended cross-encoder reranker that fixes the override-RRF failure
  mode.
- **Query-aware highlights** — sentence-level span extraction, the same
  idea behind Exa's [contents API "highlights"](https://exa.ai/docs/reference/contents).
- **Five reproducible benchmark harnesses**, full-methodology JSON
  artifacts, CI gate. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- **MCP server** (`lexa-mcp`) over stdio so any Anthropic-MCP client
  (Claude Desktop, Claude Code, Cursor, etc.) gets `search_files`,
  `index_path`, `purge_path`, and friends for free.

## How Lexa maps to Exa

| Exa concept                      | Lexa equivalent                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------|
| Instant tier (<200 ms, BM25)     | `lexa search --tier instant` — FTS5 BM25, p50 ~250 µs.                                                |
| Fast tier (~350 ms, neural)      | `lexa search --tier dense` (KNN-only) or `--tier fast` (hybrid). p50 ~9 ms.                           |
| Auto tier (~1 s, intelligent)    | `lexa search --tier auto` — query router in `classify_query`. Default tier.                           |
| Deep tier (5-60 s, agentic)      | `lexa search --tier deep` + `SearchOptions::additional_queries` for [`additionalQueries`-style](https://exa.ai/blog/exa-deep) fan-out. |
| Hybrid retrieval (BM25 + dense)  | RRF (k=60) over FTS5 BM25 and binary-quantized vector KNN, run concurrently. See [Exa: Composing a Search Engine](https://exa.ai/blog/composing-a-search-engine). |
| BM25 optimizations               | FTS5's built-in BM25 implementation; OR-of-quoted-tokens query construction with a curated stopword set. (Lexa doesn't reimplement Exa's six [posting-list compression tricks](https://exa.ai/blog/bm25-optimization) — local corpora don't justify them.) |
| Matryoshka prefix                | Nomic v1.5-Q (768d, MRL-trained at `{64, 128, 256, 512, 768}`); `vectors_bin_preview bit[256]` table for first-stage KNN. See [Exa 2.0: building a web-scale vector DB](https://exa.ai/blog/exa-api-2-0). |
| Binary quantization              | `sqlite-vec`'s `vec_quantize_binary()` and `bit[N]` columns; Hamming distance via SIMD intrinsics. 32× storage shrink. |
| Cross-encoder reranking          | `BAAI/bge-reranker-base` over top-15 fused candidates, sigmoid-blended at α = 0.7 with the RRF score. |
| Highlights / contents API        | `search.rs::highlight` — query-token-overlap-scored sentence span, ~10× LLM-token reduction vs full chunks. |
| `additionalQueries`              | `SearchOptions::additional_queries: Vec<String>`; the deep tier fans out N+1 queries, RRF-fuses them, then reranks. The bench harness includes an Ollama-backed reformulation helper. |
| LLM-as-judge eval (5-dim rubric) | `lexa-bench simpleqa` — Harness E. Scores relevance, authority, content_issues, evaluator_confidence, overall in [0, 1]. Default judge is local Ollama running `qwen3:8b`. See [Exa: Evaluating Search](https://exa.ai/blog/evals-at-exa). |

What Lexa **doesn't** clone:

- Crawl freshness — Lexa indexes static local trees, not the web.
- Websets-scale entity finding — billions of records / async enrichment
  pipelines aren't a single-binary local feature.
- Authority / domain reputation signals — those are web-graph specific.

The local-first tradeoff is what makes the latency budget viable. Exa
Fast targets <500 ms because it's reaching across a planet-scale index;
Lexa Fast hits 9 ms because everything is in SQLite next to your CPU.

## Install

```bash
cargo install --path crates/lexa-cli       # the `lexa` CLI
cargo install --path crates/lexa-mcp       # the `lexa-mcp` MCP server
```

Or run from a clone:

```bash
cargo build --workspace --release
./target/release/lexa --help
```

The first time you run a real-embedding command, fastembed downloads the
Nomic v1.5-Q ONNX (~110 MB) and the BGE-reranker-base ONNX (~280 MB) into
`./.fastembed_cache/`. Subsequent runs reuse the cache.

## CLI

```text
lexa index <path> [--db <path>]
lexa search <query> [--tier instant|dense|fast|deep|auto] [--limit N] [--json] [--db <path>]
lexa purge <path> [--db <path>]
lexa status [--db <path>]
lexa watch <path> [--db <path>]
```

Default DB is `~/.lexa/index.sqlite`. `--hash-embeddings` swaps to the
deterministic FNV-1a hash backend for tests / offline runs.

`--json` produces a stable JSON shape with `path`, `line_start`,
`line_end`, `score`, `excerpt`, and a `breakdown` object exposing the
RRF inputs, rerank score (deep only), and the routed tier (auto only).

## MCP server

Add to your MCP client config (Claude Desktop / Claude Code / Cursor):

```json
{
  "mcpServers": {
    "lexa": {
      "command": "lexa-mcp",
      "env": { "LEXA_DB": "/Users/you/.lexa/index.sqlite" }
    }
  }
}
```

Tools:

- `search_files(query, tier?, limit?)`
- `index_path(path)`
- `list_indexed_paths()`
- `purge_path(path)`
- `status()`

stderr is the only log channel; stdout is reserved for the JSON-RPC
stream so the protocol stays clean.

## Library

```toml
[dependencies]
lexa-core = "0.1"
```

```rust
use lexa_core::{open, EmbeddingConfig, SearchOptions};

let mut db = open("/tmp/lexa.sqlite", EmbeddingConfig::default())?;
db.index_path("/path/to/repo")?;

let hits = db.search(&SearchOptions::new("hybrid retrieval implementation"))?;
for hit in hits {
    println!("{}:{}-{}  {:.4}  {}", hit.path, hit.line_start, hit.line_end, hit.score, hit.excerpt);
}
```

The default `SearchOptions` uses the `auto` tier and limit 10. Set
`tier: SearchTier::Deep` and populate `additional_queries` for Exa-style
multi-query deep search.

## Obsidian — point Codex (or any MCP client) at your vault

`lexa-obsidian` is the vault-aware sibling of `lexa`. It strips YAML
frontmatter before embedding, parses Obsidian-specific syntax
(`[[wikilinks]]`, `#tags`, `^block-ids`, `![[embeds]]`), and exposes a
note-shaped MCP tool surface.

### Install

```bash
cargo install --path crates/lexa-obsidian
```

That installs two binaries: the `lexa-obsidian` CLI and the
`lexa-obsidian-mcp` stdio server.

### CLI

```text
lexa-obsidian --vault <path> index
lexa-obsidian --vault <path> status
lexa-obsidian --vault <path> tags [--prefix X] [--limit N]
lexa-obsidian --vault <path> backlinks <note>
lexa-obsidian --vault <path> search <query> [--tier auto|fast|deep] [--tag X] [--folder Y] [--limit N] [--json]
lexa-obsidian --vault <path> watch
```

`--vault` falls back to `LEXA_OBSIDIAN_VAULT`. The DB defaults to
`~/.lexa/obsidian-<sha>.sqlite` so two distinct vaults never share an
index.

### MCP server (Codex CLI / Claude Desktop / Cursor / Claude Code)

```toml
# ~/.codex/config.toml
[mcp_servers.lexa-obsidian]
command = "lexa-obsidian-mcp"
env = { LEXA_OBSIDIAN_VAULT = "/Users/<you>/Documents/MyVault" }
```

Tools exposed to the agent:

| Tool             | Description                                                                                                |
|------------------|------------------------------------------------------------------------------------------------------------|
| `search_notes`   | Hybrid search with optional `tags` and `folders` filters. Same five tiers as the core Lexa engine.        |
| `find_backlinks` | List every note that links to a target note (path or filename stem).                                       |
| `list_tags`      | Top tags by usage, optional prefix filter.                                                                  |
| `get_note`       | Fetch a single note: frontmatter, body, outgoing + incoming links, tags. Optionally restricted to a block id (`^abc`). |
| `get_similar`    | Notes semantically similar to the given seed note.                                                          |
| `index_vault`    | Force a re-index.                                                                                           |
| `purge_vault`    | Drop the index for the configured vault.                                                                    |
| `vault_status`   | DB + sidecar counts plus a `needs_index` flag callers can poll.                                             |

The server **lazily indexes on the first content-bearing tool call** so
Codex doesn't appear hung at session start. For a >1000-note vault, run
`lexa-obsidian index` ahead of time so the first MCP call is instant.

### What it parses

- YAML frontmatter (`title:`, `aliases:`, `tags:` plus arbitrary custom
  fields preserved in `note_metadata.raw_json`). Frontmatter is *stripped
  before embedding* so it doesn't pollute the vector representation.
- Wiki-links: `[[Note]]`, `[[Note|Alias]]`, `[[Note#Header]]`,
  `[[Note^block-id]]`, `![[Embed]]`. Stored in `note_links`; backlinks
  are a single SQL JOIN.
- Tags: frontmatter `tags:` (string, list, or comma-string) plus inline
  `#tag` (including nested `#project/lexa`). Lowercase-normalised.
  Code fences and heading lines are correctly skipped.
- Block ids: trailing `^block-id` markers on the last line of a chunk
  are persisted into `note_blocks` and queryable through
  `get_note { block: "^abc" }`.

### Schema (sidecar, in the same SQLite file)

```sql
note_metadata (doc_id PK, title, aliases_json, raw_json)
note_links    (id PK, src_doc_id, target_name, target_path, header, block_id, alias, kind)
note_tags     (doc_id, tag, PRIMARY KEY(doc_id, tag))
note_blocks   (chunk_id PK, doc_id, block_id)
```

`ON DELETE CASCADE` rides on `documents.id`, so purging a path through
the core Lexa API cleans the sidecars automatically.

### 5-minute Codex verification

```bash
cargo install --path crates/lexa-obsidian
lexa-obsidian --vault ~/Documents/MyVault index
# Add the [mcp_servers.lexa-obsidian] block above to ~/.codex/config.toml
echo "\nLEXA_CANARY_X7Q_obsidian_test" >> ~/Documents/MyVault/Daily/2026-05-01.md
# Restart Codex, then ask:
#   > Use lexa-obsidian. Find the note containing "LEXA_CANARY_X7Q".
```

## Benchmarks

Five harnesses, fully reproducible. Numbers below are warm-state on
M-series macOS arm64, release build, real Nomic v1.5-Q. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for hardware details, full
methodology, and the date each number was measured.

### Harness A — Latency

2 000 synthetic Markdown docs, 500 iterations / tier, fixed query set:

| Tier    | p50      | p95      | p99      |
|---------|----------|----------|----------|
| instant | 245 µs   | 840 µs   | 861 µs   |
| dense   | 8.97 ms  | 9.82 ms  | 10.20 ms |
| fast    | 9.00 ms  | 9.92 ms  | 10.19 ms |
| deep    | 261 ms   | 298 ms   | 313 ms   |

Pairs with a Criterion bench (`cargo bench -p lexa-bench --bench latency`)
and a CI gate that fails if fast-tier p50 > 400 ms on shared GitHub
Actions runners.

### Harness B — BEIR retrieval quality (SciFact, 100 queries)

| Tier    | nDCG@10  | MRR@10 | Recall@100 | p50      | p95      |
|---------|----------|--------|------------|----------|----------|
| instant | 0.6560   | 0.6184 | 0.8680     | 3 ms     | 5 ms     |
| fast    | 0.6778   | 0.6395 | 0.8980     | 17 ms    | 22 ms    |
| deep    | **0.7042** | **0.6674** | 0.8360 | 2.57 s   | 2.81 s   |

Hybrid lifts BM25-only by +2.2 nDCG points; deep adds another +2.6 nDCG
on top, **eliminating the previous deep-tier regression** caused by
unbounded reranker logits overriding RRF. Beats the published BEIR BM25
SciFact baseline (~0.665) at p95 < 25 ms on the fast tier.

### Harness C — Agent quality (20 NL queries on this repo)

| Tool                 | Tier      | Correct | Accuracy | Median latency |
|----------------------|-----------|---------|----------|----------------|
| `lexa` (Nomic)       | **auto**  | 16 / 20 | **0.80** | 11 ms          |
| `lexa` (Nomic)       | fast      | 15 / 20 | 0.75     | 10 ms          |
| `grep -rE`           | external  |  0 / 20 | 0.00     |  8 ms          |

`auto` outperforms `fast` because the router sends single-identifier
queries (`vec_quantize_binary`, `LexaDb::open`) straight to BM25-only
`instant`, where exact-symbol lookups beat hybrid scoring.

### Harness D — Head-to-head against external CLIs

Wraps any external command (`grep`, `rg`, `qmd-cli`, ...) and runs the
same query set. Reports per-tool latency and match rate against expected
file paths. See `lexa-bench compare --help`.

### Harness E — SimpleQA-style LLM-as-judge

Mirrors Exa's [evaluation methodology](https://exa.ai/blog/evals-at-exa):
hand-curated factual questions, scored on the five-dim rubric (relevance,
authority, content_issues, evaluator_confidence, overall) in `[0, 1]`.

```bash
# Local-first: judge is whatever's running in Ollama.
cargo run -p lexa-bench --release -- simpleqa \
  --queries bench/simpleqa/questions.json --corpus . \
  --tier auto --judge ollama --judge-model qwen3:8b \
  --real-embeddings --json bench-results/simpleqa.json
```

A deterministic `--judge mock` backend exists for CI smoke runs that
need to verify wiring without a model download.

### Reproducers

```bash
# Harness A — latency (writes JSON, gates CI)
cargo run -p lexa-bench --release -- latency \
  --db /tmp/lexa.sqlite --docs 2000 --iterations 500 \
  --real-embeddings --json bench-results/latency-nomic.json

# Harness A — Criterion (HTML reports under target/criterion/)
cargo bench -p lexa-bench --bench latency

# Harness B — BEIR
cargo run -p lexa-bench --release -- beir scifact --download \
  --db /tmp/lexa-scifact.sqlite --real-embeddings \
  --tiers instant,dense,fast,deep --max-queries 100 \
  --json bench-results/scifact.json

# Harness C — agent (auto tier on the lexa repo)
cargo run -p lexa-bench --release -- agent \
  --queries bench/agent/queries.json --corpus . \
  --tool lexa --tier auto --real-embeddings \
  --db /tmp/lexa-agent.sqlite \
  --json bench-results/agent-auto.json

# Harness D — head-to-head with grep
cargo run -p lexa-bench --release -- compare \
  --queries bench/agent/queries.json --corpus . \
  --command "grep -rEln {query} {corpus}/crates" --label grep \
  --json bench-results/compare-grep.json

# Harness E — SimpleQA (mock judge for CI)
cargo run -p lexa-bench --release -- simpleqa \
  --queries bench/simpleqa/questions.json --corpus . \
  --tier auto --judge mock --real-embeddings \
  --json bench-results/simpleqa.json
```

## Project layout

```text
lexa/
├── Cargo.toml                     # workspace
├── README.md                      # this file
├── crates/
│   ├── lexa-core/                 # library: chunking, embed, retrieval
│   ├── lexa-cli/                  # `lexa` binary
│   ├── lexa-mcp/                  # `lexa-mcp` rmcp stdio server
│   └── lexa-bench/                # `lexa-bench` — five harnesses
├── docs/
│   ├── ARCHITECTURE.md            # this is the design doc
│   ├── BENCHMARKS.md              # full benchmark methodology
│   └── adr/000–005-*.md           # one-page decisions
├── bench/
│   ├── agent/queries.json         # 20 NL queries against this repo
│   ├── agent/SKILL.md             # full agent-loop spec (Anthropic API)
│   └── simpleqa/questions.json    # SimpleQA seed set
├── bench-results/                 # committed JSON artifacts
└── tests/fixtures/sample/         # tiny corpus for tests
```

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE] or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT] or
  <https://opensource.org/licenses/MIT>)

at your option.
