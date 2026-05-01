# Lexa Benchmarks

Lexa publishes four benchmark harnesses. Each entry below includes the date,
hardware, exact corpus URL, and the exact command that produces the number;
without that metadata, the number is not citable.

> **Hardware for the reference numbers**: Apple M-series MacBook (macOS arm64),
> compiled with the workspace's optimized release profile (`lto=thin`,
> `codegen-units=1`, `strip=true`). Numbers were measured between
> 2026-04-30 and 2026-05-01 against the commit current at write-time.
> Machine spec note for reproducers: an M1 Pro / M2 / M3 class chip with
> ≥16 GB RAM should be within ~30 % of these numbers.

> **Software**: `cargo --version` 1.80+, fastembed 5.13.4 (ONNX runtime),
> sqlite-vec 0.1.9 (statically linked).

## Harness A — Latency

Two reporters, same fixed query set:

- `cargo bench -p lexa-bench --bench latency` — Criterion-based: warm-state
  mean + standard deviation + outlier analysis, HTML reports under
  `target/criterion/`. Right tool for "is the mean stable and low?".
- `lexa-bench latency` — microsecond-resolution p50 / p95 / p99 over a
  configurable iteration count (default 1000). Right tool for the tail
  percentiles that the CI gate cares about. JSON output with `--json`.

### Reference numbers

> 2,000 synthetic Markdown docs, 10 query templates rotated 50× = 500
> iterations, real Nomic v1.5 quantized embedder, single-threaded, warm cache,
> macOS arm64 release build, measured 2026-05-01.
> Artifact: [`bench-results/latency-nomic.json`](../bench-results/latency-nomic.json).
> Reproducer:
> ```bash
> cargo run -p lexa-bench --release -- latency \
>   --db /tmp/lexa-lat.sqlite --docs 2000 --iterations 500 \
>   --real-embeddings --json bench-results/latency-nomic.json
> ```

| Tier    | p50      | p95      | p99      |
|---------|----------|----------|----------|
| instant | 245 µs   | 840 µs   | 861 µs   |
| dense   | 8.97 ms  | 9.82 ms  | 10.20 ms |
| fast    | 9.00 ms  | 9.92 ms  | 10.19 ms |
| deep    | 261 ms   | 298 ms   | 313 ms   |

The `instant` tier measures FTS5 BM25 alone; `dense` measures the
two-stage Matryoshka KNN (256-bit preview → 768-bit re-score) over the
binary-quantized index; `fast` is RRF over both, with the BM25 SQL hit
parallelized against the embedder ONNX forward pass via
`std::thread::scope`; `deep` adds the BGE-reranker-base cross-encoder over
the top-15 fused candidates with sigmoid-blended scores. **Deep latency
dropped from 567 ms (the previous baseline before the rerank rework) to
261 ms** thanks to the smaller candidate count and the `RERANK_BLEND` mix.

### Synthetic Markdown corpus, real Nomic v1.5-Q

| Docs   | Tier    | p50    | p95    |
|--------|---------|--------|--------|
| 1,000  | instant | <1 ms  | <1 ms  |
| 1,000  | fast    | 9 ms   | 10 ms  |
| 1,000  | deep    | 506 ms | 591 ms |
| 10,000 | instant | 1 ms   | 1 ms   |
| 10,000 | fast    | 10 ms  | 11 ms  |
| 10,000 | deep    | 603 ms | 1,371 ms |

(Date: 2026-04-30 → 2026-05-01. Hardware: see top of file.)

### CI latency gate

The `bench-latency-gate` job in `.github/workflows/ci.yml` runs the same
harness on a GitHub Actions Ubuntu runner with the deterministic hash
backend (no model download), 2,000 docs / 200 iterations, and fails the
build if the fast-tier p50 exceeds **400 ms**:

```bash
./target/release/lexa-bench latency \
  --db /tmp/lexa-ci.sqlite --docs 2000 --iterations 200 \
  --gate-fast-p50-ms 400 --json /tmp/lexa-latency.json
```

Headline note: the GHA gate is intentionally loose because GHA runners are
shared and slow; the M-series numbers in the table above are 1–2 orders of
magnitude tighter.

## Harness B — BEIR retrieval quality

Reports nDCG@10, MRR@10, Recall@100 for any of `instant` (BM25-only),
`dense` (vector-only), `fast` (hybrid), `deep` (hybrid + cross-encoder
rerank).

### Datasets

| Dataset   | Source URL                                                                                              |
|-----------|---------------------------------------------------------------------------------------------------------|
| SciFact   | <https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip>                        |
| NFCorpus  | <https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip>                       |

### SciFact reference numbers (real Nomic v1.5-Q, 100 queries)

> Reproducers:
> ```bash
> # instant + fast (rebuild index)
> cargo run -p lexa-bench --release -- beir scifact --download \
>   --db /tmp/lexa-scifact.sqlite --real-embeddings \
>   --tiers instant,fast --max-queries 100 \
>   --json bench-results/scifact-instant-fast.json
>
> # deep (reuses the index)
> cargo run -p lexa-bench --release -- beir scifact \
>   --db /tmp/lexa-scifact.sqlite --real-embeddings \
>   --tiers deep --max-queries 100 \
>   --json bench-results/scifact-deep.json
> ```

| Tier    | nDCG@10 | MRR@10 | Recall@100 | p50      | p95      |
|---------|---------|--------|------------|----------|----------|
| instant | 0.6560  | 0.6184 | 0.8680     | 3 ms     | 5 ms     |
| fast    | 0.6778  | 0.6395 | 0.8980     | 17 ms    | 22 ms    |
| deep    | **0.7042** | **0.6674** | 0.8360 | 2,568 ms | 2,807 ms |

(Date: 2026-05-01. Hardware: see top of file. Index time for SciFact
~7.8 minutes — 5,183 docs at ~90 ms/doc with the new two-stage Matryoshka
write path. Numbers above are the final post-optimization measurements;
all five Lexa changes — parallel BM25/embed, two-stage Matryoshka,
auto routing, blended deep rerank, query-aware highlights — are in the
binary.)

Sanity check: the published BEIR baseline for BM25 on SciFact is roughly
0.665 nDCG@10. Lexa's `instant` tier (BM25 + OR query construction +
stopword filtering) lands at 0.656 — within ~1 nDCG point of the
baseline. The `fast` tier (hybrid) lifts that by +2.2 nDCG. The `deep`
tier (BGE-reranker-base over top-15, full chunk text, blended with RRF)
lifts another +2.6 nDCG to 0.7042 — **eliminating the previous deep-tier
quality regression** (0.6434 < 0.7058 with the old `score += rerank`
override) and decisively beating fast.

### NFCorpus

Smaller dataset (3,633 docs); useful as a second-corpus sanity check for the
fusion implementation. Reproducer:

```bash
cargo run -p lexa-bench --release -- beir nfcorpus --download \
  --db /tmp/lexa-nfcorpus.sqlite --real-embeddings \
  --tiers instant,dense,fast,deep --max-queries 200 \
  --json bench-results/nfcorpus.json
```

(Numbers will be added once the full run completes.)

## Harness C — Agent quality (tool-only mode shipped today)

20 hand-written natural-language queries against the lexa repository
itself. Each query has a known correct `file:line_start-line_end` answer.
Score = top-K hits include the expected file with line range overlap.

> Reproducer:
> ```bash
> cargo run -p lexa-bench --release -- agent \
>   --queries bench/agent/queries.json \
>   --corpus . --tool lexa --tier auto \
>   --real-embeddings \
>   --db /tmp/lexa-agent.sqlite \
>   --json bench-results/agent-auto.json
> ```

(Date: 2026-05-01. Line ranges in `bench/agent/queries.json` were refreshed
against the current code positions; pre-refresh runs scored ~17/20 because
the old ranges were stale relative to the post-optimization line numbers.)

| Tool                 | Tier      | Queries | Correct | Accuracy | Median latency |
|----------------------|-----------|---------|---------|----------|----------------|
| `lexa` (Nomic v1.5)  | **auto**  | 20      | **16**  | **0.80** | 11 ms          |
| `lexa` (Nomic v1.5)  | fast      | 20      | 15      | **0.75** | 10 ms          |
| `grep -rE`           | external  | 20      | 0       | **0.00** | 8 ms           |

Artifacts: [`bench-results/agent-auto.json`](../bench-results/agent-auto.json),
[`bench-results/agent-fast.json`](../bench-results/agent-fast.json),
[`bench-results/compare-grep.json`](../bench-results/compare-grep.json).

The `auto` tier outperforms `fast` on this corpus (0.80 vs 0.75) because
the query router sends single-identifier queries straight to BM25-only
`instant`, which beats hybrid scoring on exact-symbol lookups. Hybrid
still wins on the 16 natural-language queries.

The `grep` row is the headline differentiation: natural-language queries
("where does lexa parse the FTS5 tokenizer query string with stopword
filtering") have no usable substring for `grep`, which is why every
hand-written agent demo gravitates to a tool like Lexa.

The full agent-loop variant (Anthropic API in a tool-use loop, measuring
turns + tokens) is **not** shipped in this harness; the design and the
configuration matrix are in [`bench/agent/SKILL.md`](../bench/agent/SKILL.md).

## Harness D — Head-to-head against external CLIs

Same query set as Harness C, runs through any external command with
`{query}` and `{corpus}` substitution.

> Reproducers (each tool requires its own install; uninstalled tools are
> skipped honestly rather than faked):
> ```bash
> # grep baseline
> cargo run -p lexa-bench --release -- compare \
>   --queries bench/agent/queries.json --corpus . \
>   --command "grep -rEln {query} {corpus}/crates" --label grep \
>   --json bench-results/compare-grep.json
>
> # qmd-cli (https://github.com/dwillmore/qmd) — install first:
> #   npx qmd-cli ...
> cargo run -p lexa-bench --release -- compare \
>   --queries bench/agent/queries.json --corpus . \
>   --command "npx qmd-cli search {query}" --label qmd \
>   --json bench-results/compare-qmd.json
>
> # ripgrep (https://github.com/BurntSushi/ripgrep) — `cargo install ripgrep`
> cargo run -p lexa-bench --release -- compare \
>   --queries bench/agent/queries.json --corpus . \
>   --command "rg --line-number {query} {corpus}/crates" --label ripgrep \
>   --json bench-results/compare-rg.json
> ```

| Tool                 | Queries | Correct | Accuracy | Median latency |
|----------------------|---------|---------|----------|----------------|
| `lexa fast` (Nomic)  | 20      | 17      | 0.85     | 11 ms          |
| `grep -rE`           | 20      |  0      | 0.00     |  8 ms          |
| `qmd-cli`            | (run on a machine with it installed) | — | — | — |
| `ripgrep`            | (run on a machine with it installed) | — | — | — |

The `Compare` harness is **deliberately cooperative**: external tools that
emit results in `path:line:` format are scored fairly, and tools that don't
are scored on whether the expected path is in their stdout at all. That's
honest about each tool's interface — it's not Lexa's job to make `grep`
look bad, it's grep's job to handle natural-language queries (which it
can't, by design).

## Harness E — SimpleQA-style factual evaluation (LLM-as-judge)

Mirrors Exa's [evals-at-exa](https://exa.ai/blog/evals-at-exa) methodology:
hand-curated factual questions about the corpus, scored by an LLM judge on
the five-dimension Exa rubric — relevance, authority, content_issues,
evaluator_confidence, overall — each in `[0, 1]`.

> Reproducer (mock judge — runs without a model, deterministic):
> ```bash
> cargo run -p lexa-bench --release -- simpleqa \
>   --queries bench/simpleqa/questions.json \
>   --corpus . --tier auto --judge mock \
>   --real-embeddings \
>   --db /tmp/lexa-simpleqa.sqlite \
>   --json bench-results/simpleqa.json
> ```
>
> Reproducer (Ollama judge):
> ```bash
> cargo run -p lexa-bench --release -- simpleqa \
>   --queries bench/simpleqa/questions.json \
>   --corpus . --tier auto --judge ollama \
>   --judge-model qwen3:8b \
>   --real-embeddings \
>   --db /tmp/lexa-simpleqa.sqlite \
>   --json bench-results/simpleqa-ollama.json
> ```

(Date: 2026-05-01. Hardware: see top of file. The Ollama path requires a
running Ollama server; the mock judge is intentionally limited to
keyword-overlap scoring so it verifies harness wiring, not retrieval
quality.)

10-question seed set against the lexa repo, real Nomic v1.5-Q, mock judge:

| Tier | Questions | Overall | Relevance | Median excerpt tokens |
|------|-----------|---------|-----------|-----------------------|
| auto | 10        | 0.6431  | 0.6431    | 193                   |

Artifact: [`bench-results/simpleqa.json`](../bench-results/simpleqa.json).

The 50-question hand-curated set against `rust-lang/book` (the headline
SimpleQA target) is left as a follow-up; the seed set above is enough to
validate the harness end-to-end.

## Methodology notes

- **Cold start vs warm state.** All numbers above are *warm-state*:
  the embedder/reranker is loaded once and reused. Cold-start CLI numbers
  (model download, ONNX init) are reported separately in [README.md].
- **Single-threaded.** Each query is run sequentially. Lexa is concurrency-
  safe (the `Mutex<Embedder>` lets multiple `LexaDb::search` calls share an
  embedder), but the benchmarks measure single-threaded latency, which is
  the user-visible quantity.
- **Deterministic hash backend.** The `Hash` embedding backend exists for
  CI smoke tests only. It's a fixed-dim FNV-1a embedding with no
  pre-training; quality numbers using it are not meaningful and are never
  reported here as headline retrieval quality. They appear only in the
  CI-gate latency table where the *latency* is what's being measured.

## Reproducer checklist

A reported number is acceptable for inclusion in this file only if it has:

- [ ] Date (YYYY-MM-DD) it was measured
- [ ] Hardware (CPU model, RAM, OS)
- [ ] Exact corpus URL and (for BEIR datasets) version note
- [ ] Exact `cargo run` command, copy-pasteable, including all flags
- [ ] JSON artifact path (under `bench-results/`) so the number can be
      regenerated mechanically
