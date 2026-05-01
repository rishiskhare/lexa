# Lexa-Obsidian threat model

A pragmatic, plain-language threat model. Lexa-Obsidian is a *local*
read-only search engine for your Obsidian vault. The threat surface is
deliberately small.

## What Lexa-Obsidian does

| Action                              | Initiator                      | Confined to     |
|-------------------------------------|--------------------------------|-----------------|
| Read `.md` / `.txt` / etc files     | The CLI / MCP server, on demand| Vault directory |
| Read frontmatter, tags, wiki-links  | Indexer                        | Vault directory |
| Write to a SQLite file              | Indexer                        | `~/.lexa/`      |
| Spawn `lexa-obsidian-mcp`           | Codex / Claude Desktop / etc.  | The MCP client  |
| Download Nomic + BGE ONNX models    | First run only, from Hugging Face | `~/.cache/fastembed` |

## What Lexa-Obsidian *does not* do

- **No write access to the vault.** The MCP server has no `create_note`,
  `edit_note`, `delete_note`, or `move_note` tool. There is no code
  path that opens a vault file in write mode.
- **No execution of user-supplied subprocesses.** The MCP server only
  spawns the `lexa-obsidian-mcp` binary itself, never an arbitrary
  command from a tool argument. No shell, no eval, no code-execution
  surface.
- **No telemetry, no analytics, no reporting endpoint.** Grep the
  source for `https://`: the only outbound URLs are Hugging Face model
  fetches in `fastembed` and the BEIR dataset URL in `lexa-bench`
  (only used by the bench harness, never by `lexa-obsidian`).
- **No credential storage.** No API keys, no auth tokens, nothing
  to leak.

## Network footprint

After first-run model downloads, **zero outbound traffic**. Verify:

```bash
# Run for ten minutes of normal Lexa usage in another terminal:
sudo tcpdump -i any -nn 'host huggingface.co or host hf.co' -w /tmp/lexa-net.pcap
sudo tcpdump -r /tmp/lexa-net.pcap | wc -l   # expect 0 once the cache is hot
```

For a hard offline guarantee:

```bash
lexa-obsidian models prefetch                # populate the cache while online
# … unplug network …
lexa-obsidian --vault ~/Vault index          # works fully offline
```

## What the index file contains

`~/.lexa/obsidian-<sha>.sqlite` is a regular SQLite database. Columns:

- Document path (absolute, on your machine).
- Chunk text (verbatim from your notes).
- Binary-quantized vector embeddings (768 bits per chunk).
- Wiki-link source/target metadata.
- Tag set per note.
- Frontmatter as JSON (in `note_metadata.raw_json`).

If your notes are sensitive, treat this file the same way you treat
the vault folder itself. `chmod 600 ~/.lexa/*.sqlite` if your machine
has multiple users.

To verify the schema yourself:

```bash
sqlite3 ~/.lexa/obsidian-<sha>.sqlite '.schema'
sqlite3 ~/.lexa/obsidian-<sha>.sqlite 'select count(*) from documents'
```

## What an attacker who controls the MCP client can do

The MCP server trusts its caller. If a malicious model invokes
`search_notes` with a query string that's a prompt-injection attack,
Lexa returns the matching note text — that's exactly what
`search_notes` is supposed to do. The malicious behaviour, if any, is
on the client side, not on Lexa's side.

What Lexa cannot do for you: *prevent* a model from then exfiltrating
the returned note text by hallucinating a tool call to a write-side
MCP server. That's a property of your *MCP client* configuration, not
of Lexa. Audit which MCP servers your Codex / Claude Desktop / etc.
session has access to.

## Reporting a security issue

Open a private security advisory at
<https://github.com/rishiskhare/lexa/security/advisories/new>.
For non-security bugs, use the regular issue tracker.
