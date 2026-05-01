#!/usr/bin/env bash
# Lexa-Obsidian installer.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rishiskhare/lexa/main/scripts/install.sh | sh
#
# Behaviour, in order of preference:
#   1. Detect the host platform and download a prebuilt tarball from
#      the latest GitHub Release. Fastest path — no compilation.
#   2. If `cargo` is on PATH, fall back to `cargo install lexa-obsidian`
#      from crates.io. Builds from source; works on every Rust target
#      (notably Linux ARM64, which has no prebuilt today).
#   3. Otherwise, point the user at https://rustup.rs/.
#
# After install, prints the canonical next steps.

set -euo pipefail

REPO="rishiskhare/lexa"
RELEASES="https://github.com/${REPO}/releases"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
err() { printf '\033[31m%s\033[0m\n' "$*" >&2; }

bold "lexa-obsidian installer"
echo

# Detect the host's release-tarball triple. Returns empty string if
# we don't ship a prebuilt for this platform.
detect_target() {
    local os arch
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)
    case "$os/$arch" in
        darwin/arm64)        echo "aarch64-apple-darwin" ;;
        darwin/x86_64)       echo "x86_64-apple-darwin" ;;
        linux/x86_64)        echo "x86_64-unknown-linux-gnu" ;;
        # Linux ARM64 has no prebuilt yet — fall through to cargo.
        *)                   echo "" ;;
    esac
}

install_dir() {
    if [ -n "${LEXA_INSTALL_DIR:-}" ]; then
        echo "$LEXA_INSTALL_DIR"
        return
    fi
    if [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
        echo "$HOME/.local/bin"
        return
    fi
    echo "$HOME/.cargo/bin"
}

target=$(detect_target)

# 1. Prebuilt tarball.
if [ -n "$target" ] && command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1; then
    # Resolve the latest tag. Falls through to cargo install if the API
    # call fails (rate limit, transient network).
    tag=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
        | grep '"tag_name"' \
        | head -n1 \
        | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)
    if [ -n "$tag" ]; then
        version=${tag#v}
        url="${RELEASES}/download/${tag}/lexa-${version}-${target}.tar.gz"
        dir=$(install_dir)
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        bold "→ Downloading $url"
        if curl -fSL "$url" -o "$tmp/lexa.tar.gz"; then
            tar -xzf "$tmp/lexa.tar.gz" -C "$tmp"
            extracted="$tmp/lexa-${version}-${target}"
            install -d "$dir"
            install -m 0755 "$extracted/lexa" "$extracted/lexa-mcp" \
                "$extracted/lexa-obsidian" "$extracted/lexa-obsidian-mcp" "$dir/"
            bold "✓ installed to $dir"
            case ":$PATH:" in
                *":$dir:"*) ;;
                *)
                    warn "  $dir is not on your PATH. Add the following line to your shell rc:"
                    warn "    export PATH=\"$dir:\$PATH\""
                    ;;
            esac
            INSTALLED=1
        fi
    fi
fi

# 2. cargo install fallback (works for Linux ARM64 + anywhere we don't
#    ship a prebuilt).
if [ -z "${INSTALLED:-}" ]; then
    if command -v cargo >/dev/null 2>&1; then
        bold "→ No prebuilt for this platform. Falling back to cargo install lexa-obsidian (builds from source)."
        cargo install lexa-obsidian --locked
        bold "✓ installed via cargo (binaries in ~/.cargo/bin)"
    else
        err "Neither a prebuilt release nor cargo was found."
        err
        err "  1. Install Rust + cargo:  https://rustup.rs/"
        err "  2. Re-run this script."
        err
        err "Or grab a release tarball directly:"
        err "  ${RELEASES}"
        exit 1
    fi
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
