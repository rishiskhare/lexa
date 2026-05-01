# Harness C — Agent Quality Benchmark

## Goal

Score how well a search tool helps an LLM answer real, natural-language
questions about a real codebase. The headline question: "would replacing
`grep -r` with this tool actually save the agent turns and tokens?"

## What ships today (tool-only mode)

`lexa-bench agent` runs every query in `queries.json` through a single tool
(`lexa`, `grep`, `qmd-cli`, …) and scores correctness as
`(top-K hits include the expected file, with line range overlap)`.

This is not the full agent loop. It is the same retrieval bedrock the agent
loop would build on, and it isolates retrieval quality from agent variance.

```bash
# lexa, hash backend (deterministic, no model download)
cargo run -p lexa-bench --release -- agent \
  --queries bench/agent/queries.json \
  --corpus . \
  --tool lexa \
  --tier fast

# lexa, real Nomic embeddings
cargo run -p lexa-bench --release -- agent \
  --queries bench/agent/queries.json \
  --corpus . \
  --tool lexa \
  --tier fast \
  --real-embeddings

# grep baseline
cargo run -p lexa-bench --release -- agent \
  --queries bench/agent/queries.json \
  --corpus . \
  --tool grep
```

The corpus for the bundled query set is the lexa repository itself, so the
benchmark is fully self-contained. To benchmark on a third-party codebase,
write your own `queries.json` and point `--corpus` at it.

## Query set design

`queries.json` is a JSON array of:

```json
{
  "query": "natural language question",
  "expected_path": "relative/path/from/corpus.rs",
  "expected_line_range": [start, end]
}
```

The 20 queries shipped today were hand-written against this repo, covering
all four crates and all major retrieval components (FTS query construction,
RRF fusion, Matryoshka truncation, MCP tool definitions, CLI watch loop,
benchmark harnesses themselves). Each answer is a real `path:line_start-line_end`
that exists at the commit being benchmarked.

## What the next step (full agent loop) looks like

The full Harness C run feeds each query to Claude Sonnet (or any tool-using
model) in a tool-use loop where the model can:

1. Call the search tool one or more times.
2. Read returned chunks.
3. Decide whether it has enough context to answer.

The harness then scores three things per query:

- **Correctness** — does the model's final answer cite the expected
  file/line range?
- **Turns** — how many tool calls did it take?
- **Tokens** — total input + output tokens for the run.

Three configurations are compared head-to-head:

- `(a)` `grep -r` only
- `(b)` `lexa-mcp` (the `search_files` MCP tool)
- `(c)` `qmd-cli` (or any other compared tool)

This requires a working `ANTHROPIC_API_KEY` and access to the agent runtime.
The runner is **not implemented** in this harness because it depends on an
external API and is non-trivial to make reproducible offline. The query set
and the tool-only scoring layer above are the pieces that *are* shipped, so
the agent loop only needs to add the API call and accumulate counts.

## Acceptance criteria for the headline result

When the full agent loop runs, the headline acceptance criterion is:

> **lexa-mcp configuration finishes more queries correctly in fewer turns
> than `grep -r`** on the same query set, on a public Rust codebase.

That's the demo for Show HN. Without that, "lexa is faster than X" is just
a microbenchmark.
