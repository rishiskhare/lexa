//! YAML frontmatter parsing for Obsidian notes.
//!
//! Obsidian notes optionally start with `---\n…\n---\n`. Below the
//! terminator is the markdown body; the frontmatter itself carries
//! metadata (`title`, `aliases`, `tags`, custom fields) that should be
//! indexed in the `note_metadata` sidecar but **not** mixed into the
//! body's embedding.

use serde::Deserialize;
use serde_yaml::Value;
use std::collections::BTreeMap;

/// Parsed Obsidian frontmatter. Typed convenience fields plus the raw
/// mapping so callers can recover any custom keys.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Frontmatter {
    pub title: Option<String>,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
    /// Any frontmatter keys not consumed by the typed fields above.
    pub raw: BTreeMap<String, Value>,
}

/// Detect and parse a leading YAML frontmatter block.
///
/// Returns `(frontmatter, body, body_byte_offset)`. When the file has
/// no frontmatter or the YAML fails to parse, returns
/// `(Frontmatter::default(), full_text, 0)` so the caller's chunker
/// still sees the entire file.
pub fn parse(text: &str) -> (Frontmatter, &str, usize) {
    let Some(after_open) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (Frontmatter::default(), text, 0);
    };
    // Find the closing `---` on its own line.
    let mut search_from = 0usize;
    let close = loop {
        let Some(rel) = after_open[search_from..].find("---") else {
            return (Frontmatter::default(), text, 0);
        };
        let abs = search_from + rel;
        // Closing fence must start at column 0 of a line.
        let prev_is_newline = abs == 0 || matches!(after_open.as_bytes()[abs - 1], b'\n');
        // It must end with a line break or EOF so frontmatter "---" ≠ body "---inline".
        let after_fence = abs + 3;
        let ends_line = after_fence == after_open.len()
            || matches!(after_open.as_bytes()[after_fence], b'\n' | b'\r');
        if prev_is_newline && ends_line {
            break abs;
        }
        search_from = abs + 1;
    };
    let yaml = &after_open[..close];

    let parsed = match serde_yaml::from_str::<RawFrontmatter>(yaml) {
        Ok(parsed) => parsed,
        Err(_) => return (Frontmatter::default(), text, 0),
    };

    // Body starts after the closing `---` and the line break that follows.
    let body_start_within_after_open = {
        let mut idx = close + 3;
        if let Some(&b) = after_open.as_bytes().get(idx) {
            if b == b'\r' {
                idx += 1;
            }
        }
        if let Some(&b) = after_open.as_bytes().get(idx) {
            if b == b'\n' {
                idx += 1;
            }
        }
        idx
    };
    let prefix_len = text.len() - after_open.len();
    let body_offset = prefix_len + body_start_within_after_open;
    let body = &text[body_offset..];

    let fm = parsed.into_frontmatter();
    (fm, body, body_offset)
}

/// Resolve the displayed note title in the documented priority order:
/// frontmatter `title:` → first `# H1` in the body → file stem.
pub fn resolve_title(fm: &Frontmatter, body: &str, file_stem: &str) -> String {
    if let Some(t) = &fm.title {
        if !t.trim().is_empty() {
            return t.trim().to_string();
        }
    }
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let title = rest.trim();
            if !title.is_empty() {
                return title.to_string();
            }
        }
    }
    file_stem.to_string()
}

#[derive(Debug, Default, Deserialize)]
struct RawFrontmatter {
    #[serde(default)]
    title: Option<String>,
    #[serde(default, alias = "alias")]
    aliases: Option<Value>,
    #[serde(default, alias = "tag")]
    tags: Option<Value>,
    #[serde(flatten)]
    rest: BTreeMap<String, Value>,
}

impl RawFrontmatter {
    fn into_frontmatter(self) -> Frontmatter {
        let aliases = string_list(self.aliases);
        let tags = string_list(self.tags)
            .into_iter()
            .map(|t| t.trim_start_matches('#').to_string())
            .filter(|t| !t.is_empty())
            .collect();
        let mut raw = self.rest;
        // Strip the few keys we hoist into typed fields out of `raw`
        // so callers don't see them twice. `serde(flatten)` already
        // excludes the typed fields, but we double-check.
        for k in ["title", "aliases", "alias", "tags", "tag"] {
            raw.remove(k);
        }
        Frontmatter {
            title: self.title,
            aliases,
            tags,
            raw,
        }
    }
}

fn string_list(value: Option<Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        Value::String(s) => s
            .split([',', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
        Value::Sequence(seq) => seq
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s),
                Value::Number(n) => Some(n.to_string()),
                Value::Bool(b) => Some(b.to_string()),
                _ => None,
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_frontmatter_returns_full_body() {
        let text = "# Heading\n\nbody";
        let (fm, body, off) = parse(text);
        assert_eq!(fm, Frontmatter::default());
        assert_eq!(body, text);
        assert_eq!(off, 0);
    }

    #[test]
    fn parses_basic_frontmatter() {
        let text = "---\ntitle: Hello\ntags: [a, b]\naliases:\n  - HelloWorld\n  - Hi\n---\n# Body\n\ntext";
        let (fm, body, off) = parse(text);
        assert_eq!(fm.title.as_deref(), Some("Hello"));
        assert_eq!(fm.tags, vec!["a", "b"]);
        assert_eq!(fm.aliases, vec!["HelloWorld", "Hi"]);
        assert!(body.starts_with("# Body"));
        assert_eq!(&text[off..], body);
    }

    #[test]
    fn handles_string_aliases_field() {
        let text = "---\nalias: Foo, Bar\n---\nbody";
        let (fm, _body, _off) = parse(text);
        assert_eq!(fm.aliases, vec!["Foo", "Bar"]);
    }

    #[test]
    fn malformed_yaml_falls_back_to_full_body() {
        let text = "---\ntitle: [unterminated\n---\nbody";
        let (fm, body, off) = parse(text);
        assert_eq!(fm, Frontmatter::default());
        assert_eq!(body, text);
        assert_eq!(off, 0);
    }

    #[test]
    fn closing_fence_must_be_line_anchored() {
        // The "---" inside the YAML value must not terminate the block.
        let text = "---\ntitle: \"a---b\"\nfoo: bar\n---\nbody";
        let (fm, body, _off) = parse(text);
        assert_eq!(fm.title.as_deref(), Some("a---b"));
        assert_eq!(body, "body");
    }

    #[test]
    fn keeps_unknown_keys_in_raw() {
        let text = "---\ntitle: T\ncreated: 2026-05-01\n---\nbody";
        let (fm, _body, _off) = parse(text);
        assert!(fm.raw.contains_key("created"));
    }

    #[test]
    fn resolve_title_priority() {
        let stem = "2026-05-01";
        assert_eq!(
            resolve_title(
                &Frontmatter {
                    title: Some("Frontmatter Title".into()),
                    ..Frontmatter::default()
                },
                "# H1\nbody",
                stem
            ),
            "Frontmatter Title"
        );
        assert_eq!(
            resolve_title(&Frontmatter::default(), "# H1 Title\nbody", stem),
            "H1 Title"
        );
        assert_eq!(
            resolve_title(&Frontmatter::default(), "no headings here\n", stem),
            stem
        );
    }
}
