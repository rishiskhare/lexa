---
title: Hybrid Retrieval
tags: [research, retrieval]
aliases: [BM25 + dense, hybrid search]
---

# Hybrid Retrieval

The combination of lexical (BM25) and dense (embedding KNN) retrieval,
fused with [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)
at k=60. Used by [[Project Alpha]] and (eventually) [[Project Beta]].

## Why hybrid beats either alone

- **BM25** wins on exact-symbol queries: function names, error codes,
  literal strings.
- **Dense** wins on natural-language paraphrases: "where does the rate
  limiter back off when redis is down" → matches a chunk that talks
  about "fallback" without the literal word "back off".
- **Hybrid + RRF** captures both sets of wins by union-ranking.

## The auto-tier router

The lexa engine inspects the query before retrieval:

- Exactly one token shaped like an identifier (snake_case, CamelCase,
  or `::` / `.` path) → `instant` (BM25 only).
- ≥ 6 tokens ending with `?` → `deep` (full hybrid + cross-encoder
  rerank).
- Otherwise → `fast` (hybrid + RRF, no rerank).

Empirically this beats forcing every query through the same tier. See
[[Daily 2026-04-30#Retrieval performance numbers]] for measured numbers.

## Open questions

- Does a `dense-lite` tier (256-bit preview KNN only) make sense for
  ultra-low-latency use cases? The [[Daily 2026-05-01#dense-lite-idea]]
  paragraph has the back-of-envelope.
