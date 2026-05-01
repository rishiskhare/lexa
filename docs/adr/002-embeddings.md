# ADR-002: Embeddings

## Status

Accepted.

## Decision

Production embedder is **`nomic-ai/nomic-embed-text-v1.5-q`** (768 dims,
Apache-2 licensed, ~137M params, ~10 ms inference per query on CPU).

The model is **Matryoshka Representation Learning (MRL)-trained** at canonical
dims 64 / 128 / 256 / 512 / 768, so any prefix of the embedding is a valid
embedding in the same vector space — quality stays within ~1 nDCG point at
256 dims on MTEB (Kusupati et al., NeurIPS 2022).

The model is asymmetric: queries are prefixed with `search_query: ` and
documents with `search_document: ` before encoding. Both prefixes are applied
automatically by `Embedder::embed_query` and `Embedder::embed_documents`;
calling them with the wrong/missing prefix silently drops nDCG by several
points.

The deep-tier reranker is `BAAI/bge-reranker-base` (MIT-licensed cross-encoder).

The hash backend (`LEXA_EMBEDDER=hash`) is for CI smoke tests only — it
maps tokens through FNV-1a into a fixed-dim vector.

## Why this model

- **Matryoshka.** Unlocks a future two-stage retrieval path (preview-dim KNN
  → full-dim re-score) without needing to re-index when we cross a
  scale threshold.
- **License.** Apache-2 lets the binary ship in any context.
- **fastembed-rs first-class support.** `EmbeddingModel::NomicEmbedTextV15Q`
  ships in the crate; quantized weights, ONNX backend.
- **Quality.** ~62.4 MTEB. Beats `bge-small-en-v1.5` on every benchmark we
  ran while still being CPU-friendly.

## Schema coupling

`vectors_bin` is a `vec0(embedding bit[N])` virtual table; sqlite-vec fixes
the dim at `CREATE TABLE` time. The dim is interpolated from
`embed::EMBEDDING_DIMS` into the DDL the first time a DB is opened.
Changing the model dimension in code requires re-indexing into a fresh DB.

## Future work — two-stage Matryoshka retrieval

When corpus size approaches 1M chunks, add a `vectors_bin_preview` table at
`bit[256]` populated alongside `vectors_bin`. The fast tier searches preview
first, takes top-K * 4 candidates, then re-scores Hamming distance at
`bit[768]` for the survivors. The `matryoshka_truncate` helper is already
exposed for this. Holding off until benchmarks at million-vector scale
justify the schema cost.

## References

- Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022.
- Nomic Embed Matryoshka announcement,
  <https://www.nomic.ai/news/nomic-embed-matryoshka>.
- Hugging Face MRL primer, <https://huggingface.co/blog/matryoshka>.
