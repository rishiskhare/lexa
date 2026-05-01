#!/usr/bin/env bash
# Lexa-Obsidian installer.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rishiskhare/lexa/main/scripts/install.sh | sh
#
# Behaviour, in order of preference:
#   1. If a `lexa-installer.sh` is published with the latest GitHub
#      release, hand off to it (cargo-dist's official installer; signs
#      and verifies binaries).
#   2. Otherwise, if `cargo` is on PATH, clone the repo into a temp
#      directory and `cargo install --path crates/lexa-obsidian`.
#   3. Otherwise, point the user at https://rustup.rs/.
#
# After install, prints the canonical next steps.

set -euo pipefail

REPO="rishiskhare/lexa"
GH_API="https://api.github.com/repos/${REPO}/releases/latest"
RAW="https://raw.githubusercontent.com/${REPO}/main"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
err() { printf '\033[31m%s\033[0m\n' "$*" >&2; }

bold "lexa-obsidian installer"
echo

# 1. Prebuilt release.
if command -v curl >/dev/null 2>&1; then
    INSTALLER_URL=$(curl -fsSL "${GH_API}" 2>/dev/null \
        | grep -E '"browser_download_url".*lexa-installer\.sh' \
        | head -n1 \
        | sed -E 's/.*"(https[^"]+)".*/\1/' || true)
    if [ -n "${INSTALLER_URL:-}" ]; then
        bold "→ Using the official cargo-dist installer:"
        echo "  $INSTALLER_URL"
        exec sh -c "curl -fsSL \"$INSTALLER_URL\" | sh"
    fi
fi

# 2. Source build via cargo.
if command -v cargo >/dev/null 2>&1; then
    bold "→ No prebuilt release found. Building from source via cargo…"
    if ! command -v git >/dev/null 2>&1; then
        err "git is required for the source-build path. Install git or wait for a tagged release."
        exit 1
    fi
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT
    echo "  cloning into $TMPDIR…"
    git clone --depth 1 "https://github.com/${REPO}.git" "$TMPDIR/lexa" >/dev/null
    cd "$TMPDIR/lexa"
    cargo install --path crates/lexa-obsidian --locked
    cd - >/dev/null
    bold "✓ installed"
else
    err "Neither a prebuilt release nor cargo was found."
    err
    err "  1. Install Rust + cargo:  https://rustup.rs/"
    err "  2. Re-run this script."
    err
    err "Or grab a release tarball directly:"
    err "  https://github.com/${REPO}/releases"
    exit 1
fi

echo
bold "Next steps:"
echo "  1.  lexa-obsidian setup"
echo "      Walks you through pointing a vault, pre-indexing, and writing"
echo "      Codex / Claude Desktop / Claude Code MCP config blocks."
echo
echo "  2.  Restart your MCP client (Codex, Claude Desktop, Cursor)."
echo
echo "  3.  Ask: 'what did I write about <topic>?'"
echo
echo "  Diagnose problems any time with: lexa-obsidian doctor"
