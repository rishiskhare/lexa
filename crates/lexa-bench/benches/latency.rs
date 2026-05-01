//! Harness A — Criterion-based latency benchmark.
//!
//! Indexes a deterministic synthetic corpus once, then times each tier
//! (`instant`, `fast`, `deep`) over a fixed query set. Criterion produces
//! HTML reports with mean / std-dev / outlier analysis under
//! `target/criterion/`.
//!
//! Why a synthetic corpus instead of a Linux-docs / Wikipedia submodule?
//! The submodule plan in the project README requires hosting a >50k-chunk
//! corpus alongside the repo; until that's added we use a deterministic
//! 5,000-document Markdown fixture so the benchmark is reproducible
//! everywhere `cargo bench` runs. When the real corpora land in
//! `corpus/linux-docs-50k/` and `corpus/wiki-1m/`, swap `build_corpus()` for
//! `load_corpus_from_dir()` — the rest of the harness is corpus-agnostic.
//!
//! The companion *percentile* report (p50/p95/p99 over 1000 queries with the
//! exact same query set) lives in
//! `crates/lexa-bench/src/main.rs` under the `latency` subcommand. Criterion
//! is the right tool for warm-state mean latency with statistical rigor;
//! the subcommand is the right tool for tail percentiles, which is what the
//! CI gate cares about.

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lexa_core::{open, EmbeddingBackend, EmbeddingConfig, LexaDb, SearchOptions, SearchTier};
use tempfile::TempDir;

const CORPUS_DOCS: usize = 5_000;
const QUERIES: &[&str] = &[
    "configuration validation schema",
    "hybrid lexical dense retrieval",
    "incremental file watching",
    "server tools local paths",
    "latency recall benchmark",
    "binary quantized vector search",
    "cross encoder reranker scoring",
    "matryoshka embedding truncation",
    "fts5 bm25 sparse retrieval",
    "reciprocal rank fusion top k",
];

fn build_corpus(dir: &Path) {
    for idx in 0..CORPUS_DOCS {
        let topic = match idx % 5 {
            0 => "configuration validation and schema errors",
            1 => "hybrid retrieval with lexical and dense search",
            2 => "file watching and incremental indexing",
            3 => "protocol server tools and local paths",
            _ => "benchmark latency and recall measurement",
        };
        let body = format!(
            "# Synthetic document {idx}\n\nThis fixture covers {topic}. Unique marker doc_{idx}.\n\nThe implementation keeps stable paths and line ranges for search results.\n"
        );
        std::fs::write(dir.join(format!("doc-{idx:06}.md")), body).expect("write fixture doc");
    }
}

fn build_indexed_db(temp: &TempDir) -> LexaDb {
    let corpus = temp.path().join("corpus");
    std::fs::create_dir_all(&corpus).expect("mk corpus");
    build_corpus(&corpus);
    let mut db = open(
        temp.path().join("lexa.sqlite"),
        EmbeddingConfig {
            // Hash backend keeps the benchmark deterministic and removes
            // ONNX inference variance so we measure retrieval, not the model.
            // The `latency` subcommand in main.rs runs the same harness with
            // real BGE for the published numbers.
            backend: EmbeddingBackend::Hash,
            show_download_progress: false,
        },
    )
    .expect("open db");
    db.index_path(&corpus).expect("index");
    db
}

fn bench_latency(c: &mut Criterion) {
    let temp = tempfile::tempdir().expect("tempdir");
    let db = build_indexed_db(&temp);

    let mut group = c.benchmark_group("query_latency");
    group.throughput(Throughput::Elements(1));
    // 50 samples × the configured query budget gives criterion a stable
    // distribution without making the bench slow in CI.
    group.sample_size(50);

    for tier in [SearchTier::Instant, SearchTier::Fast, SearchTier::Deep] {
        for (idx, query) in QUERIES.iter().enumerate() {
            let id = BenchmarkId::new(format!("tier_{tier}"), idx);
            group.bench_with_input(id, query, |b, q| {
                b.iter(|| {
                    let hits = db
                        .search(&SearchOptions {
                            query: (*q).to_string(),
                            tier,
                            limit: 10,
                            additional_queries: Vec::new(),
                        })
                        .expect("search");
                    // Use `hits.len()` so the optimizer cannot eliminate the
                    // call. `criterion::black_box` is appropriate too but
                    // returning `usize` from the closure communicates intent
                    // and avoids importing yet another helper.
                    hits.len()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_latency);
criterion_main!(benches);
