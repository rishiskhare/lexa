---
title: Project Alpha
tags: [project, project/lexa, status/shipped]
aliases: [Alpha]
created: 2026-04-12
shipped: 2026-05-01
---

# Project Alpha

The hybrid-retrieval Obsidian adapter. Shipped on [[Daily 2026-05-01]].

## Goals

- Frontmatter-stripped embeddings.
- Wiki-link backlink graph.
- Block-id addressable retrieval.
- MCP server for Codex / Claude Desktop / Cursor.

## Outcome

Hit all goals. The auto-tier routing was the surprise — single-
identifier queries route to BM25-only `instant` and beat the hybrid
tier on exact-symbol lookups. See [[Research/Hybrid Retrieval]].

Linked to [[Project Beta]] for the next round.
