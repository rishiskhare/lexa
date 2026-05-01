# ADR-004: Search tiers

Lexa exposes five `SearchTier` variants to mirror Exa's product surface
([Exa: Composing a Search Engine](https://exa.ai/blog/composing-a-search-engine),
[Exa 2.1](https://exa.ai/blog/exa-api-2-1)) over a local corpus.

| Lexa tier | Inspiration (Exa)         | What it runs                                             |
|-----------|---------------------------|----------------------------------------------------------|
| `instant` | Exa Instant (sub-200 ms)  | FTS5 BM25 only — no model load.                          |
| `dense`   | Exa Fast (neural)         | Two-stage Matryoshka KNN only.                           |
| `fast`    | Exa Auto / Fast (hybrid)  | BM25 + dense + RRF (k=60), parallelized via `thread::scope`. |
| `deep`    | Exa Deep                  | `fast` + BGE cross-encoder rerank (top-15, sigmoid-blended). Supports `additional_queries` à la Exa Deep's `additionalQueries`. |
| `auto`    | Exa Auto                  | Routes per query: identifier-shaped → `instant`; long question-form → `deep`; default → `fast`.   |

`auto` is the default. The router lives in
`crates/lexa-core/src/search.rs::classify_query` and is unit-tested.

The `dense` tier is exposed primarily for ablation (it lets the BEIR
harness measure dense-only nDCG separately from hybrid), but it's also a
clean product knob for callers who explicitly want pure semantic search.
