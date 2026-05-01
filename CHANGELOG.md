# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-01

Initial public release.

### Added

- **`lexa` core engine** — local hybrid retrieval: BM25 (FTS5) +
  binary-quantized 768-d Matryoshka KNN (sqlite-vec) + cross-encoder
  rerank (BGE-reranker-base). Five tiers (`instant`, `dense`, `fast`,
  `deep`, `auto`) mirroring Exa's tiered API. Sub-10 ms `fast`-tier
  latency on real Nomic v1.5-Q embeddings. See [`docs/BENCHMARKS.md`].
- **`lexa-obsidian`** — vault-aware adapter. Strips frontmatter before
  embedding, parses `[[wikilinks]]`, `#tags`, `^block-ids`, `![[embeds]]`
  into sidecar tables. Read-only on the vault; per-vault DB at
  `~/.lexa/obsidian-<sha>.sqlite`.
- **`lexa-obsidian-mcp`** — rmcp stdio MCP server with eight tools:
  `search_notes`, `find_backlinks`, `list_tags`, `get_note`,
  `get_similar`, `index_vault`, `purge_vault`, `vault_status`.
  Background indexing with non-blocking progress polling — Codex
  never appears hung at session start.
- **`lexa-obsidian setup`** — interactive bootstrap that picks a vault,
  optionally pre-indexes, writes Codex / Claude Desktop / Claude Code
  MCP config blocks idempotently, and drops `AGENTS.md` in the vault
  root so agents route note questions through Lexa without prompting.
- **`lexa-obsidian doctor`** — diagnoses every common failure mode
  (binary on PATH, vault accessible, DB built, models cached, Codex
  config block present) in one command.
- **`lexa-obsidian models prefetch`** — downloads retrieval models
  (~390 MB) ahead of time so the first MCP call is instant.
- **`--offline` / `LEXA_OFFLINE=1`** — sets `HF_HUB_OFFLINE=1` so the
  fastembed model loader refuses every network call. Useful as a
  hard-offline guarantee after `models prefetch`.
- **Five reproducible benchmark harnesses** under `lexa-bench`:
  latency (Criterion + percentile reporter), BEIR retrieval quality
  (SciFact / NFCorpus), agent-task accuracy, head-to-head against
  external CLIs, SimpleQA-style LLM-as-judge.
- **Demo vault** at `demo-vault/` so reviewers can try Lexa without
  pointing it at their own notes.
- **Six ADRs** under `docs/adr/`: name, sqlite-vec choice, embedding
  model, chunking, search tiers, MCP posture, Obsidian adapter.

### Verified

- 47 tests passing across 11 suites.
- `cargo fmt --all -- --check` clean.
- `cargo clippy --workspace --all-targets --release -- -D warnings`
  clean.
- Full SciFact BEIR run with real Nomic v1.5-Q: instant nDCG@10 0.6849,
  fast 0.7058, deep 0.6434 (deep regression documented and addressed
  in v0.2 plan).

[Unreleased]: https://github.com/rishiskhare/lexa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rishiskhare/lexa/releases/tag/v0.1.0
