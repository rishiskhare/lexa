---
title: Demo Vault
tags: [meta]
---

# Lexa demo vault

A tiny synthetic Obsidian vault you can point Lexa at without your own
data. Useful for kicking the tires on the install, the MCP server, or
the agent loop.

## How to use

```bash
lexa-obsidian --vault ./demo-vault setup
```

…then ask Codex (or Claude Desktop / Cursor) things like:

- *what did I write about retrieval latency?*
- *list my top 5 tags*
- *show me backlinks for "Project Alpha"*
- *find notes similar to "Daily 2026-04-30"*
- *expand block ^perf-block in "Daily 2026-04-30"*

Notes worth knowing about:

- See [[Project Alpha]] and [[Project Beta]] for project notes.
- See [[Daily 2026-04-30]], [[Daily 2026-05-01]] for journal entries.
- See [[Research/Hybrid Retrieval]] for a research note that links to
  multiple projects.
