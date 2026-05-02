# Lexa FAQ

## Why is the first question slow?

The first time `lexa-obsidian-mcp` starts against a fresh vault, two
things happen:

1. **Indexing.** Each `.md` file is read, frontmatter parsed, body
   chunked, and every chunk embedded with the Nomic v1.5 ONNX model.
   Throughput is ~90 ms per note on M-series CPUs — so a 5 000-note
   vault takes ~7 minutes the very first time.
2. **Model download.** First-ever invocation pulls Nomic v1.5
   (~110 MB) and BGE reranker (~280 MB) from Hugging Face. This
   happens once.

To eliminate both for your future sessions:

```bash
lexa-obsidian models prefetch                # downloads the ONNX models
lexa-obsidian --vault ~/Vault index          # runs indexing visibly with progress
```

After that, `lexa-obsidian-mcp` opens instantly — the index file under
`~/.lexa/obsidian-<sha>.sqlite` already has every note, every wiki-link,
every tag.

While indexing is in flight, the MCP server returns a fast `{indexing:
true, ...}` response on every content-bearing tool call instead of
blocking, so Codex won't appear hung.

## Will Lexa pick up notes I add or change after the MCP server is running?

Yes, **automatically**. Since v0.1.1, `lexa-obsidian-mcp` spawns a
filesystem watcher (`notify-debouncer-mini`) on startup. Notes you
add, edit, or delete in Obsidian appear in (or vanish from) search
within ~500 ms of the file save — no manual `index_vault` call needed.

The watcher debounces 500 ms of FS events (so Obsidian's per-keystroke
auto-save bursts collapse to one re-index) and the re-index is
idempotent — only chunks whose content hash changed get re-embedded.

If you'd rather run the watcher externally (e.g. via launchd /
systemd) and keep the MCP server static, set
`LEXA_OBSIDIAN_NO_WATCH=1` and run `lexa-obsidian watch` separately.

## How do I switch vaults?

```bash
lexa-obsidian setup
```

Re-run setup with the new vault path. Each vault gets its own DB
(`~/.lexa/obsidian-<sha>.sqlite`), so the old one isn't touched. To
delete the old vault's index:

```bash
sha=$(lexa-obsidian --vault /path/to/old-vault status | grep db_path | cut -d'"' -f2)
rm "$sha"
```

Or just `rm ~/.lexa/obsidian-*.sqlite` to nuke them all and start over.

## Does it support "<thing Obsidian has>"?

Today:

| Feature                                       | Status |
|-----------------------------------------------|--------|
| YAML frontmatter (title / aliases / tags)     | ✅      |
| Wiki-links: `[[Note]]`, alias, header, block  | ✅      |
| Embeds: `![[Note]]`                           | ✅ (stored as edges, not inlined into the parent's embedding) |
| Inline tags `#foo`, nested `#proj/lexa`       | ✅      |
| Block ids: `^abc-123`                         | ✅ (queryable via `get_note { block: "^abc" }`) |
| Callouts: `> [!note]`                         | ✅ (preserved as text) |
| Aliases as standalone search hits             | 🚧 (stored, but not separately indexed yet) |
| Embeds inlined into parent embedding          | 🚧 (planned phase 2) |
| Dataview blocks                               | ❌      |
| `.canvas` files                               | ❌      |
| Excalidraw drawings                           | ❌      |
| PDF annotations inside the vault              | ❌      |
| Bidirectional sync (writing notes)            | ❌ (Lexa is read-only; pair with `MarkusPfundstein/mcp-obsidian` for the write path) |

## How do I uninstall?

```bash
# Remove the binaries
cargo uninstall lexa-obsidian
# (or, if installed from a tarball, just delete them from ~/.cargo/bin)
rm ~/.cargo/bin/lexa-obsidian ~/.cargo/bin/lexa-obsidian-mcp 2>/dev/null

# Remove the DB(s) and model cache
rm -rf ~/.lexa
rm -rf ~/.cache/fastembed             # default fastembed cache location
rm -rf ./.fastembed_cache              # fallback CWD location

# Remove the AGENTS.md if you dropped one in your vault
rm ~/Documents/MyVault/AGENTS.md       # adjust path

# Remove the MCP server entry from your client config
#   ~/.codex/config.toml                          [mcp_servers.lexa-obsidian]
#   ~/.claude.json                                "mcpServers"."lexa-obsidian"
#   ~/Library/Application Support/Claude/claude_desktop_config.json (Claude Desktop)
```

## Is my data sent anywhere?

No. After the first-run model download from Hugging Face, **zero**
network calls. To verify:

```bash
sudo tcpdump -i any 'host huggingface.co or host hf.co'
# (then use Lexa for ten minutes and watch the trace stay empty)
```

The server only spawns the `lexa-obsidian-mcp` binary itself, never
user-supplied subprocesses, and is read-only on your vault. See
[`docs/adr/005-mcp.md`](adr/005-mcp.md) for the full posture.

If you need a hard guarantee that nothing reaches the network at all,
use the `lexa-obsidian models prefetch` command once on a connected
machine, then unplug the network — every Lexa command after the cache
is hot is fully offline.

## Does it work on Windows / Linux / WSL?

Prebuilt binaries on every GitHub Release tag for: macOS arm64,
macOS x86_64, Linux x86_64, Windows x86_64. The `scripts/install.sh`
one-liner picks the right tarball automatically.

**Linux ARM64**: not in the prebuilt matrix because `rmcp`'s
`transport-io` feature transitively pulls `native-tls` → `openssl-sys`,
which needs an aarch64 sysroot to cross-compile from x86_64 GitHub
Actions runners. Easy native-build path:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install lexa-obsidian
```

(That works on any Linux ARM64 host directly — Raspberry Pi, AWS
Graviton, Apple Silicon under Asahi, etc.)

## What model does it use? Can I swap?

Default: `nomic-ai/nomic-embed-text-v1.5` (quantized, 768 dims,
Matryoshka-trained at canonical prefix dims `{64, 128, 256, 512, 768}`)
+ `BAAI/bge-reranker-base`. Both Apache-2 / MIT.

There's no swap toggle today; that's a deliberate v0.1 simplification
so the SQLite schema (which embeds the model dimension into the
`vec0(embedding bit[768])` table type) stays stable. A model-swap
flow that handles re-indexing into a fresh DB is on the roadmap.

## How do I run it without "Use lexa-obsidian." in every prompt?

`lexa-obsidian setup` writes a system-prompt block into
`~/.codex/config.toml` (under `[default_session].instructions`) and
drops an `AGENTS.md` in your vault root. Both nudge MCP-aware models
to route note questions through Lexa automatically. If you skipped
those steps, run `setup` again or hand-edit those two files using the
templates in `templates/AGENTS.md`.

## My DB got into a weird state. How do I rebuild it?

```bash
rm ~/.lexa/obsidian-<sha>.sqlite        # find the path with `lexa-obsidian status`
lexa-obsidian --vault ~/Vault index
```

The sidecar tables and the lexa-core tables migrate together every
time the DB opens; deleting the file is the cleanest reset.

## What does `lexa-obsidian doctor` check?

- both binaries are findable
- vault path resolves to a directory and contains `.md` files
- the SQLite DB exists and has notes / tags / links indexed
- the ONNX model cache is hot
- `~/.codex/config.toml` has a registered `[mcp_servers.lexa-obsidian]`
  block

If any of those is wrong, you get an explicit error message naming the
exact env var or config key to fix.
