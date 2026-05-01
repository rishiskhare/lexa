# Lexa

> **Local-first hybrid retrieval for your Obsidian vault and code.** A
> single static Rust binary plus an MCP server, so Codex / Claude
> Desktop / Cursor / Claude Code can answer questions from your notes
> without anything leaving your machine.

[![Crates.io](https://img.shields.io/crates/v/lexa-obsidian?label=lexa-obsidian)](https://crates.io/crates/lexa-obsidian)
[![Crates.io](https://img.shields.io/crates/v/lexa-core?label=lexa-core)](https://crates.io/crates/lexa-core)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![CI](https://github.com/rishiskhare/lexa/actions/workflows/ci.yml/badge.svg)](https://github.com/rishiskhare/lexa/actions/workflows/ci.yml)
[![Release](https://github.com/rishiskhare/lexa/actions/workflows/release.yml/badge.svg)](https://github.com/rishiskhare/lexa/actions/workflows/release.yml)

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/rishiskhare/lexa/main/scripts/install.sh | sh
lexa-obsidian setup
# restart Codex / Claude Desktop / Cursor, then ask:
#   > what did I write about <topic>?
```

That's it. `setup` is interactive: it picks up your vault, optionally
pre-indexes it, writes the right MCP config block into
`~/.codex/config.toml` (and Claude Desktop / Claude Code if you opt
in), and drops an `AGENTS.md` in your vault root so agents route note
questions through Lexa **without** the "Use lexa-obsidian." prefix.

---

## Demo

```text
> what did I write about the rate limiter redis fallback last quarter?
```

```json
[
  {
    "path": "Daily/2026-04-30.md",
    "title": "Daily 2026-04-30",
    "score": 0.7141,
    "heading": "Followups",
    "excerpt": "redis down, switching to in-memory backoff in `acquire`. The reranker latency was acceptable at p95 of 261 ms.",
    "line_start": 12,
    "line_end": 18,
    "tags": ["daily", "project/lexa"],
    "breakdown": { "routed_to": "fast" }
  }
]
```

Want to try it without your own vault? Point Lexa at the bundled
[`demo-vault/`](demo-vault/):

```bash
lexa-obsidian --vault ./demo-vault setup --no-codex --no-agents-md
lexa-obsidian --vault ./demo-vault --hash-embeddings search "reranker latency"
```

---

## Features

| Feature                                  | What it gives you                                                                  |
|------------------------------------------|------------------------------------------------------------------------------------|
| **Hybrid retrieval**                     | BM25 (FTS5) + binary-quantized 768-d Matryoshka KNN (sqlite-vec) + cross-encoder rerank, fused with RRF (k=60). |
| **Five tiers**                           | `instant` / `dense` / `fast` / `deep` / `auto` — mirroring Exa's API. p50 ~9 ms on the fast tier with real Nomic v1.5-Q. |
| **Auto-routing**                         | Single-identifier queries → `instant` (BM25); long question-form → `deep`; default → `fast`. No "Use lexa-obsidian." needed. |
| **Obsidian-native**                      | Frontmatter stripped before embedding. Wiki-links, inline `#tags`, `^block-ids`, and `![[embeds]]` parsed into sidecar tables. |
| **MCP-first**                            | Eight tools — `search_notes`, `find_backlinks`, `list_tags`, `get_note`, `get_similar`, `index_vault`, `purge_vault`, `vault_status`. |
| **Background indexing**                  | The MCP server indexes in the background; content calls return `{indexing: true, ...}` while in-flight, so Codex never hangs. |
| **One static binary**                    | No daemon, no Python, no Docker. SQLite + `sqlite-vec` is the entire backend.      |
| **100% local**                           | First run downloads Nomic + BGE ONNX (~390 MB). After that, zero network calls. No telemetry, no API keys. |
| **`--offline` flag**                     | Sets `HF_HUB_OFFLINE=1` so fastembed refuses every network fetch — hard offline guarantee after `models prefetch`. |

---

## Use cases

| Want to...                                  | Do this                                                                |
|---------------------------------------------|------------------------------------------------------------------------|
| Ask Codex / Claude / Cursor about your vault| `lexa-obsidian setup` and ask in plain English.                        |
| Search a code repo from the CLI             | `lexa index ~/repo && lexa search "rate limiter fallback"`.            |
| Wire MCP search into a custom agent         | Use `lexa-obsidian-mcp` (rmcp stdio) or `lexa-mcp` (file-shaped).      |
| Embed retrieval into your own Rust app      | `lexa_obsidian::LexaObsidianDb::open(...)` — see the [crate docs](https://docs.rs/lexa-obsidian). |
| Try it without your own data                | `lexa-obsidian --vault ./demo-vault setup --no-codex --no-agents-md`.  |

---

## Subcommands

```text
lexa-obsidian setup            # interactive bootstrap (most users only need this)
lexa-obsidian doctor           # diagnose every common failure mode
lexa-obsidian models prefetch  # download retrieval models (~390 MB) ahead of time
lexa-obsidian --vault <path> index
lexa-obsidian --vault <path> status
lexa-obsidian --vault <path> tags [--prefix X] [--limit N]
lexa-obsidian --vault <path> backlinks <note>
lexa-obsidian --vault <path> search <query> [--tier auto|fast|deep] [--tag X] [--folder Y] [--json]
lexa-obsidian --vault <path> watch
```

`--vault` falls back to `LEXA_OBSIDIAN_VAULT`. The DB defaults to
`~/.lexa/obsidian-<sha-of-vault>.sqlite` — two distinct vaults never
share an index.

---

## Other ways to install

```bash
# Cargo (any Rust target, including Linux ARM64):
cargo install lexa-obsidian

# Homebrew (macOS) — coming soon, file an issue if you want it sooner.

# Manual: grab a tarball from
# https://github.com/rishiskhare/lexa/releases/latest
```

---

## Project layout

```text
lexa/
├── crates/
│   ├── lexa-core/        # Library: chunking, embedding, hybrid retrieval, fusion.
│   ├── lexa-cli/         # `lexa` CLI for any file tree (code, docs).
│   ├── lexa-mcp/         # `lexa-mcp` rmcp stdio server (file-shaped tools).
│   ├── lexa-obsidian/    # `lexa-obsidian` CLI + `lexa-obsidian-mcp` server.
│   └── lexa-bench/       # Five reproducible benchmark harnesses.
├── docs/
│   ├── ARCHITECTURE.md   # Crate map, schema, retrieval pipeline.
│   ├── BENCHMARKS.md     # Latency / BEIR / agent / SimpleQA numbers.
│   ├── FAQ.md            # First-run latency, vault switching, uninstall, privacy.
│   ├── THREAT_MODEL.md   # Read-only-on-vault guarantee, network footprint.
│   └── adr/              # Six one-page architecture decision records.
├── bench/                # Hand-curated query sets for the agent + SimpleQA harnesses.
├── bench-results/        # Committed JSON artifacts so every published number is reproducible.
├── demo-vault/           # 6-note synthetic vault — try Lexa without your own data.
├── templates/AGENTS.md   # Dropped into your vault root by `setup`.
└── scripts/install.sh    # `curl | sh` installer.
```

---

## Documentation

- **Architecture** — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Benchmarks** — [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): real numbers
  with hardware, exact corpus URL, and the command that produces them.
- **FAQ** — [`docs/FAQ.md`](docs/FAQ.md): first-run delay, vault
  switching, uninstall, model swap, privacy verification with `tcpdump`.
- **Threat model** — [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md):
  what Lexa does, what it does *not* do, and how to verify each claim.
- **Decision log** — [`docs/adr/`](docs/adr/): six ADRs covering name,
  storage, embeddings, chunking, search tiers, MCP posture, Obsidian
  adapter.

---

## How Lexa maps to Exa

Lexa is local-first, but the architecture follows Exa's: tiered
search, hybrid retrieval, Matryoshka prefixes, binary quantization,
query-aware highlights.

| Exa concept                      | Lexa equivalent                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------|
| Instant tier (<200 ms, BM25)     | `--tier instant` — FTS5 BM25 only, p50 ~250 µs.                                                       |
| Fast tier (neural / hybrid)      | `--tier dense` (KNN-only) or `--tier fast` (hybrid + RRF), p50 ~9 ms.                                 |
| Auto tier                        | `--tier auto` — query router in `classify_query`. Default.                                            |
| Deep tier (agentic)              | `--tier deep` + `SearchOptions::additional_queries` for [`additionalQueries`-style](https://exa.ai/blog/exa-deep) fan-out.    |
| Hybrid retrieval (BM25 + dense)  | RRF (k=60) over FTS5 BM25 and binary-quantized vector KNN, run concurrently. See [Exa: Composing a Search Engine](https://exa.ai/blog/composing-a-search-engine). |
| Matryoshka prefix                | Nomic v1.5-Q (768d, MRL-trained at `{64, 128, 256, 512, 768}`) — `vectors_bin_preview bit[256]` table. See [Exa 2.0](https://exa.ai/blog/exa-api-2-0). |
| Binary quantization              | `sqlite-vec`'s `vec_quantize_binary()` and `bit[N]` columns; SIMD Hamming distance.                   |
| Cross-encoder reranking          | `BAAI/bge-reranker-base` over top-15 fused candidates, sigmoid-blended at α = 0.7 with the RRF score.|
| Highlights / contents API        | `search.rs::highlight` — query-token-overlap-scored sentence span.                                    |
| LLM-as-judge eval                | `lexa-bench simpleqa` — Harness E. Five-dim rubric. Default judge `qwen3:8b` via local Ollama.        |

---

## Privacy

Lexa runs **entirely locally**. The only outbound network call is the
first-run download of two ONNX models (Nomic v1.5-Q ~110 MB, BGE
reranker ~280 MB) from Hugging Face. After that — zero network. No
telemetry, no analytics, no API keys.

For a hard offline guarantee, run `lexa-obsidian models prefetch` once
on a connected machine, then use `--offline` (or `LEXA_OFFLINE=1`) to
make fastembed refuse every network call. See
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for the verification
recipe (`tcpdump`-style proof) and the full posture.

---

## Development

```bash
git clone https://github.com/rishiskhare/lexa
cd lexa
cargo build --workspace --release
cargo test --workspace --release
cargo clippy --workspace --all-targets --release -- -D warnings
cargo fmt --all -- --check
```

47 tests across 11 suites, no `unsafe` outside the SQLite extension
loader, all hot paths covered by Criterion benches in
[`crates/lexa-bench/benches/`](crates/lexa-bench/benches/).

---

## Contributing

Issues and PRs welcome. The decision log under
[`docs/adr/`](docs/adr/) is the best place to start if you want to
understand the design before changing it. For larger changes, open an
issue first so the architectural fit can be discussed.

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option.
