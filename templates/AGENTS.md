# Working with this Obsidian vault

You have access to the **lexa-obsidian** MCP server, which indexes the
notes in this vault with hybrid retrieval (BM25 + dense + cross-encoder
rerank, all local).

For any question that could be answered from the user's notes —
content lookup, recall, summarisation, backlinks, tags, "find a note
about X" — call the appropriate `lexa-obsidian` tool **before**
answering from prior knowledge.

| User asks                          | Call                |
|------------------------------------|---------------------|
| "what did I write about X?"        | `search_notes`      |
| "show me note Y" / "expand block Z"| `get_note`          |
| "what links to Y?"                 | `find_backlinks`    |
| "find notes similar to Y"          | `get_similar`       |
| "what tags do I use most?"         | `list_tags`         |

Default the `tier` argument of `search_notes` to `auto`. Don't add
"Use lexa-obsidian." to your responses; the routing should be implicit.

`lexa-obsidian` is read-only on this vault. To create or edit notes,
use a separate write-side MCP server (e.g. `MarkusPfundstein/mcp-obsidian`
backed by Obsidian's Local REST API plugin).
