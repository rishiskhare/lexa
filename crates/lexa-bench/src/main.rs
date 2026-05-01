//! Lexa benchmark harness.
//!
//! This binary collects every reproducible benchmark Lexa publishes:
//!
//! - `latency` — Harness A: p50/p95/p99 over a fixed query set, microsecond
//!   resolution. Pairs with the criterion benchmark in `benches/latency.rs`
//!   (mean+stddev with HTML reports). Has a `--gate-fast-p50-ms` flag for CI.
//! - `beir` — Harness B: nDCG@10, MRR@10, Recall@100 on a BEIR dataset
//!   (scifact, nfcorpus). Supports BM25-only, dense-only, hybrid, and
//!   hybrid+rerank ablations.
//! - `agent` — Harness C: runs a hand-curated query set against a tool
//!   (`lexa`, `grep`, etc.) and scores correctness based on whether the
//!   expected file is in the top result. The scaffolding for the Anthropic
//!   agent loop is in `bench/agent/SKILL.md`; this subcommand handles the
//!   tool-only mode shipped today.
//! - `compare` — Harness D: runs the same query set through any external
//!   command and reports latency + match rate. Used for head-to-head
//!   comparisons (qmd, sift, shinpr/mcp-local-rag).
//! - `synthetic` — smoke / development latency check (kept).
//! - `fixture` — smoke / development correctness check (kept).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::process::Command as Proc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use lexa_core::{open, EmbeddingBackend, EmbeddingConfig, SearchOptions, SearchTier};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "lexa-bench", version, about = "Lexa benchmark harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Harness A. Microsecond-resolution latency report over a fixed query
    /// set. Use `--gate-fast-p50-ms N` to fail the process if the fast tier
    /// p50 exceeds N (used in CI).
    Latency {
        #[arg(long)]
        db: PathBuf,
        #[arg(long, default_value_t = 5_000)]
        docs: usize,
        #[arg(long, default_value_t = 1_000)]
        iterations: usize,
        #[arg(long)]
        real_embeddings: bool,
        /// Optional CI gate: fail with non-zero exit if `tier=fast` p50 in
        /// milliseconds exceeds this value.
        #[arg(long)]
        gate_fast_p50_ms: Option<u64>,
        /// Optional JSON output path for machine consumption.
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Harness B. BEIR retrieval-quality benchmark.
    /// Datasets: `scifact`, `nfcorpus`.
    Beir {
        dataset: String,
        #[arg(long)]
        download: bool,
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        real_embeddings: bool,
        #[arg(long, default_value_t = 200)]
        max_queries: usize,
        /// Comma-separated list of tiers to evaluate. Examples:
        ///   --tiers instant,dense,fast,deep
        #[arg(long, value_delimiter = ',')]
        tiers: Vec<String>,
        /// Generate `additional_queries` reformulations via Ollama and
        /// fan them out at the Deep tier (Exa Deep `additionalQueries`).
        /// Default 0 (off); 2 is the canonical Exa Deep choice.
        #[arg(long, default_value_t = 0)]
        expand: usize,
        /// Override Ollama base URL. Default `http://localhost:11434`.
        #[arg(long)]
        ollama_url: Option<String>,
        /// Ollama model used for reformulation. Default `qwen3:8b` —
        /// Qwen3 dominates the 7-8B class on reasoning + structured-output
        /// benchmarks (HumanEval 76.0, native JSON-schema enforcement).
        /// Swap to `qwen3:4b` on RAM-constrained machines (rivals
        /// Qwen2.5-72B per Alibaba's own evals).
        #[arg(long)]
        ollama_model: Option<String>,
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Harness C. Agent-quality benchmark over a hand-curated query set.
    /// The `tool-only` mode shipped today runs each query through the
    /// configured tool (lexa, grep, etc.) and scores correctness by
    /// checking whether the expected `file:line_start-line_end` answer
    /// appears in the top-K result. The `agent` mode (Anthropic tool loop)
    /// is scaffolded in `bench/agent/SKILL.md`; not invoked here.
    Agent {
        /// Query set path. JSON: `[{ "query": "...", "expected_path": "...",
        /// "expected_line_range": [start, end] }, ...]`.
        #[arg(long)]
        queries: PathBuf,
        /// Path to the corpus to index (or that the external tool will scan).
        #[arg(long)]
        corpus: PathBuf,
        /// `lexa` (built-in) or any external command (`grep`, `qmd-cli`, …).
        #[arg(long, default_value = "lexa")]
        tool: String,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        real_embeddings: bool,
        #[arg(long, default_value = "fast")]
        tier: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Harness D. Head-to-head against an external CLI.
    /// Wraps an arbitrary command string with `{query}` substitution and
    /// reports per-query latency and a regex match against the expected
    /// answer. Use this to compare lexa vs `grep -rE`, qmd-cli, etc.
    Compare {
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        /// Shell command template. `{query}` is substituted with the query
        /// text, `{corpus}` with the corpus path. Example:
        ///   --command "grep -rEn '{query}' {corpus}"
        #[arg(long)]
        command: String,
        #[arg(long, default_value = "external")]
        label: String,
        #[arg(long)]
        json: Option<PathBuf>,
    },

    Fixture {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        real_embeddings: bool,
    },

    Synthetic {
        #[arg(long)]
        db: PathBuf,
        #[arg(long, default_value_t = 1000)]
        docs: usize,
        #[arg(long, default_value_t = 100)]
        queries: usize,
        #[arg(long)]
        real_embeddings: bool,
    },

    /// Harness E. SimpleQA-style factual evaluation with LLM-as-judge.
    /// Indexes the corpus, runs each question through the configured tier,
    /// concatenates the top-K highlights, and asks an LLM to score the
    /// retrieved evidence on the Exa five-dimension rubric (relevance,
    /// authority, content issues, evaluator confidence, overall).
    Simpleqa {
        /// Question set path. JSON: `[{ "question": "...",
        /// "expected_fact": "...", "expected_path": "..." }, ...]`.
        #[arg(long)]
        queries: PathBuf,
        /// Path to the corpus to index for the eval (e.g. a clone of
        /// `rust-lang/book`).
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        real_embeddings: bool,
        #[arg(long, default_value = "auto")]
        tier: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Judge backend. `ollama` (default) talks to a local Ollama
        /// server. `mock` produces deterministic scores for tests.
        #[arg(long, default_value = "ollama")]
        judge: String,
        /// Judge model. For `ollama`, defaults to `qwen3:8b` (best small
        /// reasoning model with strict JSON-schema enforcement). For
        /// constrained-RAM hosts, `qwen3:4b` is the recommended swap.
        #[arg(long)]
        judge_model: Option<String>,
        /// Override Ollama base URL.
        #[arg(long)]
        ollama_url: Option<String>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Latency {
            db,
            docs,
            iterations,
            real_embeddings,
            gate_fast_p50_ms,
            json,
        } => run_latency(
            db,
            docs,
            iterations,
            embedding_config(real_embeddings),
            gate_fast_p50_ms,
            json.as_deref(),
        ),
        Command::Beir {
            dataset,
            download,
            db,
            real_embeddings,
            max_queries,
            tiers,
            expand,
            ollama_url,
            ollama_model,
            json,
        } => run_beir(
            &dataset,
            download,
            db,
            embedding_config(real_embeddings),
            max_queries,
            &tiers,
            expand,
            ollama_url.as_deref(),
            ollama_model.as_deref(),
            json.as_deref(),
        ),
        Command::Agent {
            queries,
            corpus,
            tool,
            db,
            real_embeddings,
            tier,
            limit,
            json,
        } => run_agent(AgentArgs {
            queries_path: &queries,
            corpus: &corpus,
            tool: &tool,
            db_path: db,
            config: embedding_config(real_embeddings),
            tier: &tier,
            limit,
            json_path: json.as_deref(),
        }),
        Command::Compare {
            queries,
            corpus,
            command,
            label,
            json,
        } => run_compare(&queries, &corpus, &command, &label, json.as_deref()),
        Command::Fixture {
            db,
            corpus,
            real_embeddings,
        } => run_fixture(db, corpus, embedding_config(real_embeddings)),
        Command::Synthetic {
            db,
            docs,
            queries,
            real_embeddings,
        } => run_synthetic(db, docs, queries, embedding_config(real_embeddings)),
        Command::Simpleqa {
            queries,
            corpus,
            db,
            real_embeddings,
            tier,
            top_k,
            judge,
            judge_model,
            ollama_url,
            json,
        } => run_simpleqa(SimpleqaArgs {
            queries_path: &queries,
            corpus: &corpus,
            db_path: db,
            config: embedding_config(real_embeddings),
            tier: &tier,
            top_k,
            judge: &judge,
            judge_model: judge_model.as_deref(),
            ollama_url: ollama_url.as_deref(),
            json_path: json.as_deref(),
        }),
    }
}

// =====================================================================
// Harness A — Latency.
// =====================================================================

const LATENCY_QUERIES: &[&str] = &[
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

fn run_latency(
    db_path: PathBuf,
    docs: usize,
    iterations: usize,
    config: EmbeddingConfig,
    gate_fast_p50_ms: Option<u64>,
    json_path: Option<&Path>,
) -> anyhow::Result<()> {
    let corpus = std::env::temp_dir().join(format!("lexa-latency-{}-{docs}", std::process::id()));
    fs::create_dir_all(&corpus)?;
    write_synthetic_corpus(&corpus, docs)?;

    let mut db = open(db_path, config)?;
    db.index_path(&corpus)?;

    let mut report = LatencyReport {
        iterations,
        docs,
        tiers: Vec::new(),
    };

    for tier in [
        SearchTier::Instant,
        SearchTier::Dense,
        SearchTier::Fast,
        SearchTier::Deep,
    ] {
        let mut samples_us = Vec::with_capacity(iterations);
        for i in 0..iterations {
            let query = LATENCY_QUERIES[i % LATENCY_QUERIES.len()];
            let start = Instant::now();
            db.search(&SearchOptions {
                query: query.to_string(),
                tier,
                limit: 10,
                additional_queries: Vec::new(),
            })?;
            samples_us.push(start.elapsed().as_micros() as u64);
        }
        samples_us.sort_unstable();
        let p50 = percentile(&samples_us, 0.50);
        let p95 = percentile(&samples_us, 0.95);
        let p99 = percentile(&samples_us, 0.99);
        println!("tier={tier} iterations={iterations} p50_us={p50} p95_us={p95} p99_us={p99}");
        report.tiers.push(LatencyTierReport {
            tier: tier.to_string(),
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
        });
    }

    if let Some(path) = json_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }

    if let Some(gate_ms) = gate_fast_p50_ms {
        let fast = report
            .tiers
            .iter()
            .find(|t| t.tier == "fast")
            .ok_or_else(|| anyhow::anyhow!("no fast-tier sample in report"))?;
        let fast_p50_ms = fast.p50_us / 1000;
        if fast_p50_ms > gate_ms {
            anyhow::bail!(
                "CI gate failed: fast-tier p50={fast_p50_ms}ms exceeds budget {gate_ms}ms"
            );
        }
        eprintln!("CI gate ok: fast-tier p50={fast_p50_ms}ms <= {gate_ms}ms");
    }

    Ok(())
}

#[derive(Serialize)]
struct LatencyReport {
    iterations: usize,
    docs: usize,
    tiers: Vec<LatencyTierReport>,
}

#[derive(Serialize)]
struct LatencyTierReport {
    tier: String,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
}

// =====================================================================
// Harness B — BEIR retrieval quality.
// =====================================================================

#[allow(clippy::too_many_arguments)]
fn run_beir(
    dataset: &str,
    download: bool,
    db_path: PathBuf,
    config: EmbeddingConfig,
    max_queries: usize,
    tier_filter: &[String],
    expand: usize,
    ollama_url: Option<&str>,
    ollama_model: Option<&str>,
    json_path: Option<&Path>,
) -> anyhow::Result<()> {
    let info = beir_dataset(dataset)?;
    if download {
        prepare_beir(&info)?;
    }
    let root = beir_root();
    if !root.join(info.subdir).exists() {
        anyhow::bail!("{dataset} data is missing; rerun with --download");
    }

    let docs = materialize_beir_docs(&root, &info)?;
    let mut db = open(db_path, config)?;
    let index_start = Instant::now();
    db.index_path(&docs)?;
    println!("index_seconds={:.2}", index_start.elapsed().as_secs_f32());

    let queries = load_queries(&root.join(info.subdir).join("queries.jsonl"))?;
    let qrels = load_qrels(&root.join(info.subdir).join("qrels/test.tsv"))?;

    let tiers: Vec<SearchTier> = if tier_filter.is_empty() {
        vec![
            SearchTier::Instant,
            SearchTier::Dense,
            SearchTier::Fast,
            SearchTier::Deep,
        ]
    } else {
        tier_filter
            .iter()
            .map(|name| name.parse::<SearchTier>())
            .collect::<Result<_, _>>()?
    };

    let mut report = BeirReport {
        dataset: dataset.to_string(),
        tiers: Vec::new(),
    };

    for tier in tiers {
        let mut latencies = Vec::new();
        let mut ndcg = Vec::new();
        let mut mrr = Vec::new();
        let mut recall = Vec::new();

        for (query_id, relevant) in qrels.iter().take(max_queries) {
            let Some(query) = queries.get(query_id) else {
                continue;
            };
            // For Deep tier with `expand > 0`, generate Ollama-backed
            // reformulations and fan them out alongside the main query.
            // Skipped on tiers that don't support `additional_queries`,
            // and skipped silently if Ollama is unreachable.
            let extras = if expand > 0 && tier == SearchTier::Deep {
                ollama_reformulate(query, expand, ollama_url, ollama_model).unwrap_or_default()
            } else {
                Vec::new()
            };
            let start = Instant::now();
            let hits = db.search(&SearchOptions {
                query: query.clone(),
                tier,
                limit: 100,
                additional_queries: extras,
            })?;
            latencies.push(start.elapsed().as_millis() as u64);
            let returned: Vec<String> = hits
                .iter()
                .filter_map(|hit| {
                    Path::new(&hit.path)
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(str::to_string)
                })
                .collect();
            recall.push(recall_at(&returned, relevant, 100));
            ndcg.push(ndcg_at(&returned, relevant, 10));
            mrr.push(mrr_at(&returned, relevant, 10));
        }

        latencies.sort_unstable();
        let row = BeirTierReport {
            tier: tier.to_string(),
            queries: ndcg.len(),
            ndcg_at_10: mean(&ndcg),
            mrr_at_10: mean(&mrr),
            recall_at_100: mean(&recall),
            p50_ms: percentile(&latencies, 0.50),
            p95_ms: percentile(&latencies, 0.95),
        };
        println!("tier={}", row.tier);
        println!("queries={}", row.queries);
        println!("ndcg_at_10={:.4}", row.ndcg_at_10);
        println!("mrr_at_10={:.4}", row.mrr_at_10);
        println!("recall_at_100={:.4}", row.recall_at_100);
        println!("p50_ms={}", row.p50_ms);
        println!("p95_ms={}", row.p95_ms);
        report.tiers.push(row);
    }

    if let Some(path) = json_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

struct BeirInfo {
    name: &'static str,
    subdir: &'static str,
    url: &'static str,
}

fn beir_dataset(name: &str) -> anyhow::Result<BeirInfo> {
    match name {
        "scifact" => Ok(BeirInfo {
            name: "scifact",
            subdir: "scifact",
            url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        }),
        "nfcorpus" => Ok(BeirInfo {
            name: "nfcorpus",
            subdir: "nfcorpus",
            url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        }),
        other => anyhow::bail!("unsupported BEIR dataset '{other}'"),
    }
}

fn prepare_beir(info: &BeirInfo) -> anyhow::Result<()> {
    let root = beir_root();
    fs::create_dir_all(&root)?;
    let zip_path = root.join(format!("{}.zip", info.name));
    if !zip_path.exists() {
        let mut response = ureq::get(info.url).call()?.into_reader();
        let mut file = fs::File::create(&zip_path)?;
        io::copy(&mut response, &mut file)?;
    }
    if !root.join(info.subdir).exists() {
        let file = fs::File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&root)?;
    }
    Ok(())
}

fn beir_root() -> PathBuf {
    std::env::var_os("LEXA_BEIR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("lexa-beir"))
}

fn materialize_beir_docs(root: &Path, info: &BeirInfo) -> anyhow::Result<PathBuf> {
    let docs = root.join(format!("{}-docs", info.name));
    if docs.exists() {
        // Reuse already-materialized corpus; the indexer's content-hash skip
        // takes care of avoiding redundant work.
        return Ok(docs);
    }
    fs::create_dir_all(&docs)?;
    let corpus = fs::File::open(root.join(info.subdir).join("corpus.jsonl"))?;
    for line in io::BufReader::new(corpus).lines() {
        let row: CorpusRow = serde_json::from_str(&line?)?;
        let body = format!("# {}\n\n{}", row.title.unwrap_or_default(), row.text);
        fs::write(docs.join(format!("{}.md", safe_id(&row.id))), body)?;
    }
    Ok(docs)
}

#[derive(Serialize)]
struct BeirReport {
    dataset: String,
    tiers: Vec<BeirTierReport>,
}

#[derive(Serialize)]
struct BeirTierReport {
    tier: String,
    queries: usize,
    ndcg_at_10: f32,
    mrr_at_10: f32,
    recall_at_100: f32,
    p50_ms: u64,
    p95_ms: u64,
}

// =====================================================================
// Harness C — Agent quality (tool-only mode).
// =====================================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
struct AgentQuery {
    query: String,
    expected_path: String,
    /// Inclusive line range \[start, end\] for the expected answer.
    expected_line_range: [i64; 2],
}

#[derive(Serialize)]
struct AgentReport {
    tool: String,
    tier: String,
    queries: usize,
    correct: usize,
    accuracy: f32,
    median_latency_ms: u64,
}

struct AgentArgs<'a> {
    queries_path: &'a Path,
    corpus: &'a Path,
    tool: &'a str,
    db_path: Option<PathBuf>,
    config: EmbeddingConfig,
    tier: &'a str,
    limit: usize,
    json_path: Option<&'a Path>,
}

fn run_agent(args: AgentArgs<'_>) -> anyhow::Result<()> {
    let AgentArgs {
        queries_path,
        corpus,
        tool,
        db_path,
        config,
        tier,
        limit,
        json_path,
    } = args;
    let queries: Vec<AgentQuery> = serde_json::from_str(&fs::read_to_string(queries_path)?)?;
    let tier_parsed: SearchTier = tier.parse()?;
    let mut latencies = Vec::with_capacity(queries.len());
    let mut correct = 0usize;

    let lexa_db = if tool == "lexa" {
        let db_path = db_path.unwrap_or_else(|| std::env::temp_dir().join("lexa-agent.sqlite"));
        let mut db = open(db_path, config)?;
        db.index_path(corpus)?;
        Some(db)
    } else {
        None
    };

    for q in &queries {
        let start = Instant::now();
        let hits: Vec<(String, i64, i64)> = if let Some(db) = &lexa_db {
            db.search(&SearchOptions {
                query: q.query.clone(),
                tier: tier_parsed,
                limit,
                additional_queries: Vec::new(),
            })?
            .into_iter()
            .map(|hit| (hit.path, hit.line_start, hit.line_end))
            .collect()
        } else {
            run_external_tool(tool, &q.query, corpus, limit)?
        };
        latencies.push(start.elapsed().as_millis() as u64);

        let expected = canonical_path(corpus.join(&q.expected_path).as_path())
            .unwrap_or_else(|| corpus.join(&q.expected_path).to_string_lossy().into_owned());
        if hits.iter().any(|(path, line_start, line_end)| {
            let same_file = canonical_path(Path::new(path))
                .map(|p| p == expected)
                .unwrap_or_else(|| Path::new(path).ends_with(&q.expected_path));
            let overlaps =
                !(*line_end < q.expected_line_range[0] || *line_start > q.expected_line_range[1]);
            same_file && overlaps
        }) {
            correct += 1;
        }
    }

    latencies.sort_unstable();
    let report = AgentReport {
        tool: tool.to_string(),
        tier: tier.to_string(),
        queries: queries.len(),
        correct,
        accuracy: correct as f32 / queries.len().max(1) as f32,
        median_latency_ms: percentile(&latencies, 0.50),
    };
    println!("tool={}", report.tool);
    println!("tier={}", report.tier);
    println!("queries={}", report.queries);
    println!("correct={}", report.correct);
    println!("accuracy={:.3}", report.accuracy);
    println!("median_latency_ms={}", report.median_latency_ms);

    if let Some(path) = json_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_external_tool(
    tool: &str,
    query: &str,
    corpus: &Path,
    limit: usize,
) -> anyhow::Result<Vec<(String, i64, i64)>> {
    // Generic shell-out used by the agent harness when `--tool` is something
    // other than `lexa`. The convention is that the tool prints lines of the
    // form `path:line:` (grep-compatible). We parse those into hits.
    let output = Proc::new("sh")
        .arg("-c")
        .arg(format!(
            "{tool} -nE {query:?} {corpus:?} 2>/dev/null | head -n {limit}",
            corpus = corpus.display()
        ))
        .output()?;
    let mut hits = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        // Best-effort: `path:line:rest`. A miss falls through to no-match.
        let mut parts = line.splitn(3, ':');
        if let (Some(path), Some(line_no), Some(_)) = (parts.next(), parts.next(), parts.next()) {
            if let Ok(ln) = line_no.parse::<i64>() {
                hits.push((path.to_string(), ln, ln));
            }
        }
    }
    Ok(hits)
}

fn canonical_path(path: &Path) -> Option<String> {
    fs::canonicalize(path)
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

// =====================================================================
// Harness D — Head-to-head against an external CLI.
// =====================================================================

fn run_compare(
    queries_path: &Path,
    corpus: &Path,
    command: &str,
    label: &str,
    json_path: Option<&Path>,
) -> anyhow::Result<()> {
    let queries: Vec<AgentQuery> = serde_json::from_str(&fs::read_to_string(queries_path)?)?;
    let mut latencies = Vec::with_capacity(queries.len());
    let mut correct = 0usize;

    for q in &queries {
        let cmd = command
            .replace("{query}", &shell_escape(&q.query))
            .replace("{corpus}", &shell_escape(&corpus.to_string_lossy()));
        let start = Instant::now();
        let output = Proc::new("sh").arg("-c").arg(&cmd).output()?;
        latencies.push(start.elapsed().as_millis() as u64);
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Score: top-K output contains the expected path. Line-range matching
        // is best-effort here — many external tools don't print line numbers
        // in a uniform format.
        if stdout.contains(&q.expected_path) {
            correct += 1;
        }
    }

    latencies.sort_unstable();
    let report = AgentReport {
        tool: label.to_string(),
        tier: "external".to_string(),
        queries: queries.len(),
        correct,
        accuracy: correct as f32 / queries.len().max(1) as f32,
        median_latency_ms: percentile(&latencies, 0.50),
    };
    println!("tool={}", report.tool);
    println!("queries={}", report.queries);
    println!("correct={}", report.correct);
    println!("accuracy={:.3}", report.accuracy);
    println!("median_latency_ms={}", report.median_latency_ms);

    if let Some(path) = json_path {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn shell_escape(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

// =====================================================================
// Smoke benchmarks (kept for development).
// =====================================================================

fn run_fixture(db_path: PathBuf, corpus: PathBuf, config: EmbeddingConfig) -> anyhow::Result<()> {
    let mut db = open(db_path, config)?;
    db.index_path(&corpus)?;
    let queries = [
        ("config validation function", "config"),
        ("hybrid local search", "search"),
        ("project architecture", "architecture"),
    ];
    let mut latencies = Vec::new();
    let mut correct = 0usize;
    for tier in [SearchTier::Instant, SearchTier::Fast, SearchTier::Deep] {
        for (query, expected) in queries {
            let start = Instant::now();
            let hits = db.search(&SearchOptions {
                query: query.to_string(),
                tier,
                limit: 10,
                additional_queries: Vec::new(),
            })?;
            latencies.push(start.elapsed().as_millis() as u64);
            if hits
                .first()
                .map(|hit| hit.excerpt.to_ascii_lowercase().contains(expected))
                .unwrap_or(false)
            {
                correct += 1;
            }
        }
    }
    latencies.sort_unstable();
    println!("fixture_accuracy={:.3}", correct as f32 / 9.0);
    println!("p50_ms={}", percentile(&latencies, 0.50));
    println!("p95_ms={}", percentile(&latencies, 0.95));
    Ok(())
}

fn run_synthetic(
    db_path: PathBuf,
    docs: usize,
    queries: usize,
    config: EmbeddingConfig,
) -> anyhow::Result<()> {
    let corpus = std::env::temp_dir().join(format!("lexa-synthetic-{}-{docs}", std::process::id()));
    fs::create_dir_all(&corpus)?;
    write_synthetic_corpus(&corpus, docs)?;

    let mut db = open(db_path, config)?;
    let indexed = db.index_path(&corpus)?;
    println!("indexed_documents={indexed}");
    for tier in [SearchTier::Instant, SearchTier::Fast, SearchTier::Deep] {
        let mut latencies = Vec::new();
        for idx in 0..queries {
            let query = LATENCY_QUERIES[idx % LATENCY_QUERIES.len()];
            let start = Instant::now();
            db.search(&SearchOptions {
                query: query.to_string(),
                tier,
                limit: 10,
                additional_queries: Vec::new(),
            })?;
            latencies.push(start.elapsed().as_millis() as u64);
        }
        latencies.sort_unstable();
        println!("tier={tier}");
        println!("queries={queries}");
        println!("p50_ms={}", percentile(&latencies, 0.50));
        println!("p95_ms={}", percentile(&latencies, 0.95));
    }
    Ok(())
}

fn write_synthetic_corpus(dir: &Path, docs: usize) -> anyhow::Result<()> {
    for idx in 0..docs {
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
        fs::write(dir.join(format!("doc-{idx:06}.md")), body)?;
    }
    Ok(())
}

// =====================================================================
// Harness E — SimpleQA-style factual evaluation with LLM-as-judge.
// =====================================================================
//
// Mirrors Exa's SimpleQA / MS-Marco-adapted methodology described in their
// "How to Evaluate Exa Search" post. For each hand-curated factual
// question, run the configured tier, concatenate the top-K highlights as
// retrieved evidence, and ask a judge LLM to score the evidence on five
// dimensions: relevance, authority, content_issues, evaluator_confidence,
// overall. Aggregate dimension means and an `overall_score` headline.
//
// We pick a JSON-output prompt that requests scores in [0, 1] so the
// judge contract is uniform across Ollama / OpenAI-compatible / Anthropic
// backends. The Ollama path is the local-first default; `mock` is a
// deterministic backend for tests so the harness stays runnable in CI
// without network or models.

#[derive(Debug, Deserialize, Serialize, Clone)]
struct SimpleqaQuestion {
    question: String,
    expected_fact: String,
    #[serde(default)]
    expected_path: Option<String>,
}

#[derive(Debug, Default, Serialize, Clone)]
struct JudgeScore {
    relevance: f32,
    authority: f32,
    content_issues: f32,
    evaluator_confidence: f32,
    overall: f32,
}

#[derive(Serialize)]
struct SimpleqaPerQuery {
    question: String,
    expected_fact: String,
    routed_to: Option<String>,
    score: JudgeScore,
    excerpt_token_count: usize,
}

#[derive(Serialize)]
struct SimpleqaReport {
    tier: String,
    judge: String,
    judge_model: String,
    questions: usize,
    relevance_mean: f32,
    authority_mean: f32,
    content_issues_mean: f32,
    evaluator_confidence_mean: f32,
    overall_mean: f32,
    median_excerpt_tokens: u64,
    per_query: Vec<SimpleqaPerQuery>,
}

struct SimpleqaArgs<'a> {
    queries_path: &'a Path,
    corpus: &'a Path,
    db_path: Option<PathBuf>,
    config: EmbeddingConfig,
    tier: &'a str,
    top_k: usize,
    judge: &'a str,
    judge_model: Option<&'a str>,
    ollama_url: Option<&'a str>,
    json_path: Option<&'a Path>,
}

#[allow(clippy::too_many_arguments)]
fn run_simpleqa(args: SimpleqaArgs<'_>) -> anyhow::Result<()> {
    let SimpleqaArgs {
        queries_path,
        corpus,
        db_path,
        config,
        tier,
        top_k,
        judge,
        judge_model,
        ollama_url,
        json_path,
    } = args;

    let questions: Vec<SimpleqaQuestion> =
        serde_json::from_str(&fs::read_to_string(queries_path)?)?;
    let tier_parsed: SearchTier = tier.parse()?;
    let db_path = db_path.unwrap_or_else(|| std::env::temp_dir().join("lexa-simpleqa.sqlite"));
    let mut db = open(db_path, config)?;
    db.index_path(corpus)?;

    let model = judge_model.unwrap_or("qwen3:8b");

    let mut per_query = Vec::with_capacity(questions.len());
    let mut relevance = Vec::with_capacity(questions.len());
    let mut authority = Vec::with_capacity(questions.len());
    let mut issues = Vec::with_capacity(questions.len());
    let mut confidence = Vec::with_capacity(questions.len());
    let mut overall = Vec::with_capacity(questions.len());
    let mut excerpt_tokens = Vec::with_capacity(questions.len());

    for q in &questions {
        let hits = db.search(&SearchOptions {
            query: q.question.clone(),
            tier: tier_parsed,
            limit: top_k,
            additional_queries: Vec::new(),
        })?;
        let evidence: String = hits
            .iter()
            .enumerate()
            .map(|(i, hit)| {
                format!(
                    "[{i}] {}:{}-{}\n{}",
                    hit.path, hit.line_start, hit.line_end, hit.excerpt
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let token_estimate = evidence.split_whitespace().count();
        excerpt_tokens.push(token_estimate as u64);

        let score = match judge {
            "mock" => mock_judge(&q.question, &q.expected_fact, &evidence),
            "ollama" => ollama_judge(&q.question, &q.expected_fact, &evidence, ollama_url, model)
                .unwrap_or_else(|err| {
                    eprintln!("ollama judge failed for {:?}: {err}", q.question);
                    JudgeScore::default()
                }),
            other => anyhow::bail!("unknown judge backend: {other}"),
        };
        relevance.push(score.relevance);
        authority.push(score.authority);
        issues.push(score.content_issues);
        confidence.push(score.evaluator_confidence);
        overall.push(score.overall);

        let routed_to = hits
            .first()
            .and_then(|h| h.breakdown.routed_to)
            .map(|t| t.to_string());

        per_query.push(SimpleqaPerQuery {
            question: q.question.clone(),
            expected_fact: q.expected_fact.clone(),
            routed_to,
            score,
            excerpt_token_count: token_estimate,
        });
    }

    excerpt_tokens.sort_unstable();
    let report = SimpleqaReport {
        tier: tier.to_string(),
        judge: judge.to_string(),
        judge_model: model.to_string(),
        questions: questions.len(),
        relevance_mean: mean(&relevance),
        authority_mean: mean(&authority),
        content_issues_mean: mean(&issues),
        evaluator_confidence_mean: mean(&confidence),
        overall_mean: mean(&overall),
        median_excerpt_tokens: percentile(&excerpt_tokens, 0.50),
        per_query,
    };

    println!(
        "tier={} judge={} questions={} overall_mean={:.4} relevance_mean={:.4} median_excerpt_tokens={}",
        report.tier,
        report.judge,
        report.questions,
        report.overall_mean,
        report.relevance_mean,
        report.median_excerpt_tokens,
    );

    if let Some(path) = json_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

/// Deterministic judge for tests / CI. Scores high when the expected fact's
/// significant tokens appear in the evidence and low otherwise. Doesn't
/// require a network or a model; intentionally not a quality signal — it
/// only verifies that the harness wires up correctly.
fn mock_judge(_question: &str, expected_fact: &str, evidence: &str) -> JudgeScore {
    let expected_tokens: HashSet<String> = expected_fact
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| s.len() > 3)
        .map(|s| s.to_ascii_lowercase())
        .collect();
    if expected_tokens.is_empty() {
        return JudgeScore::default();
    }
    let evidence_lower = evidence.to_ascii_lowercase();
    let hits = expected_tokens
        .iter()
        .filter(|t| evidence_lower.contains(t.as_str()))
        .count();
    let ratio = hits as f32 / expected_tokens.len() as f32;
    JudgeScore {
        relevance: ratio,
        authority: 0.5,
        content_issues: 1.0 - (1.0 - ratio).min(0.5),
        evaluator_confidence: 0.8,
        overall: ratio,
    }
}

/// Ollama-backed LLM-as-judge. Sends a single prompt requesting JSON in the
/// Exa five-dim rubric, parses, and falls back to default scores if the
/// model returns malformed JSON.
fn ollama_judge(
    question: &str,
    expected_fact: &str,
    evidence: &str,
    base_url: Option<&str>,
    model: &str,
) -> anyhow::Result<JudgeScore> {
    let url = format!(
        "{}/api/generate",
        base_url.unwrap_or("http://localhost:11434")
    );
    let prompt = format!(
        "You are an evaluator for a search engine. Score the retrieved \
         evidence against the expected answer on five dimensions, each in \
         [0, 1]. Output **only** a JSON object with keys: relevance, \
         authority, content_issues, evaluator_confidence, overall. No \
         prose, no markdown, no preamble.\n\n\
         - relevance: Does the evidence address the question?\n\
         - authority: Is the evidence from an authoritative section of \
           the corpus (definitions, type signatures, public API) versus a \
           test or comment?\n\
         - content_issues: 1.0 means no issues; 0.0 means the evidence is \
           wrong, misleading, or off-topic.\n\
         - evaluator_confidence: How confident are you in your scoring?\n\
         - overall: Aggregate score for the retrieval quality.\n\n\
         Question: {question}\n\
         Expected fact: {expected_fact}\n\
         Evidence:\n{evidence}\n\n\
         JSON:"
    );
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "format": "json",
        "options": { "temperature": 0.0, "num_predict": 256 }
    });
    let resp: serde_json::Value = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(60))
        .send_json(body)?
        .into_json()?;
    let text = resp
        .get("response")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("ollama: missing `response` field"))?;
    let parsed: serde_json::Value = serde_json::from_str(text)?;
    Ok(JudgeScore {
        relevance: parsed
            .get("relevance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        authority: parsed
            .get("authority")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        content_issues: parsed
            .get("content_issues")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        evaluator_confidence: parsed
            .get("evaluator_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        overall: parsed
            .get("overall")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
    })
}

// =====================================================================
// Shared helpers.
// =====================================================================

fn embedding_config(real_embeddings: bool) -> EmbeddingConfig {
    EmbeddingConfig {
        backend: if real_embeddings {
            EmbeddingBackend::FastEmbed
        } else {
            EmbeddingBackend::Hash
        },
        show_download_progress: real_embeddings,
    }
}

fn load_queries(path: &Path) -> anyhow::Result<HashMap<String, String>> {
    let file = fs::File::open(path)?;
    let mut out = HashMap::new();
    for line in io::BufReader::new(file).lines() {
        let row: QueryRow = serde_json::from_str(&line?)?;
        out.insert(row.id, row.text);
    }
    Ok(out)
}

fn load_qrels(path: &Path) -> anyhow::Result<Vec<(String, HashSet<String>)>> {
    let file = fs::File::open(path)?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for (idx, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if idx == 0 && line.contains("query-id") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 && parts[2].parse::<i32>().unwrap_or(0) > 0 {
            out.entry(parts[0].to_string())
                .or_default()
                .insert(safe_id(parts[1]));
        }
    }
    let mut rows: Vec<_> = out.into_iter().collect();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(rows)
}

fn recall_at(returned: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = returned
        .iter()
        .take(k)
        .filter(|doc_id| relevant.contains(*doc_id))
        .count();
    hits as f32 / relevant.len() as f32
}

fn ndcg_at(returned: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    let dcg = returned
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, doc_id)| relevant.contains(*doc_id))
        .map(|(idx, _)| 1.0 / ((idx + 2) as f32).log2())
        .sum::<f32>();
    let ideal = (0..usize::min(k, relevant.len()))
        .map(|idx| 1.0 / ((idx + 2) as f32).log2())
        .sum::<f32>();
    if ideal == 0.0 {
        0.0
    } else {
        dcg / ideal
    }
}

fn mrr_at(returned: &[String], relevant: &HashSet<String>, k: usize) -> f32 {
    returned
        .iter()
        .take(k)
        .enumerate()
        .find_map(|(idx, doc_id)| {
            if relevant.contains(doc_id) {
                Some(1.0 / (idx + 1) as f32)
            } else {
                None
            }
        })
        .unwrap_or(0.0)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(values: &[u64], pct: f32) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let idx = ((values.len() as f32 - 1.0) * pct).round() as usize;
    values[idx]
}

/// Generate `n` query reformulations via a local Ollama server.
///
/// Mirrors Exa Deep's `additionalQueries` (auto 2–3): we ask the LLM for
/// short paraphrases that should retrieve the same documents but with
/// different lexical surface forms. Falls back to `Err` (which the caller
/// turns into "no expansion") if Ollama is unreachable, the response is
/// malformed, or the model returns fewer than 1 line.
fn ollama_reformulate(
    query: &str,
    n: usize,
    base_url: Option<&str>,
    model: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let url = format!(
        "{}/api/generate",
        base_url.unwrap_or("http://localhost:11434")
    );
    let model = model.unwrap_or("qwen3:8b");
    let prompt = format!(
        "Rewrite this search query in {n} different ways. Only output the \
         rewritten queries, one per line, no numbering, no explanation, \
         no quotes. Each rewrite should preserve the semantics but vary \
         the phrasing or the keywords.\n\nQuery: {query}\n\nRewrites:"
    );
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": { "temperature": 0.2, "num_predict": 200 }
    });
    let resp: serde_json::Value = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(15))
        .send_json(body)?
        .into_json()?;
    let text = resp
        .get("response")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("ollama: missing `response` field"))?;
    let lines: Vec<String> = text
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches([
                    '-', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ')', ' ',
                ])
                .trim()
                .to_string()
        })
        .filter(|line| !line.is_empty() && line.len() <= 256)
        .take(n)
        .collect();
    if lines.is_empty() {
        anyhow::bail!("ollama returned no usable lines");
    }
    Ok(lines)
}

fn safe_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Deserialize)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    title: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}
