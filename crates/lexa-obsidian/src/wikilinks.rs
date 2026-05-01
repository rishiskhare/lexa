//! Obsidian wiki-link and embed extraction.
//!
//! Recognises:
//! - `[[Note]]`
//! - `[[Note|Alias]]`
//! - `[[Note#Header]]`
//! - `[[Note^block-id]]`
//! - `[[Note#Header|Alias]]`, `[[Note^block|Alias]]`
//! - `![[Note]]` (transclusion / embed)
//!
//! Resolution against the actual `documents.path` rows is the indexer's
//! job; this module is purely text → tuples.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkKind {
    Link,
    Embed,
}

impl LinkKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkKind::Link => "link",
            LinkKind::Embed => "embed",
        }
    }
}

/// One wiki-link or embed extracted from a note body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wikilink {
    pub target_name: String,
    pub header: Option<String>,
    pub block_id: Option<String>,
    pub alias: Option<String>,
    pub kind: LinkKind,
}

/// Scan the note body and return every wiki-link / embed.
///
/// The body is expected to be already-stripped of frontmatter; passing
/// the full file is harmless but matches inside the YAML block.
pub fn extract(body: &str) -> Vec<Wikilink> {
    let re = pattern();
    re.captures_iter(body)
        .filter_map(|cap| {
            let target = cap.get(2)?.as_str().trim();
            if target.is_empty() {
                return None;
            }
            let kind = if cap.get(1).map(|m| !m.is_empty()).unwrap_or(false) {
                LinkKind::Embed
            } else {
                LinkKind::Link
            };
            Some(Wikilink {
                target_name: target.to_string(),
                header: cap.get(3).map(|m| m.as_str().trim().to_string()),
                block_id: cap.get(4).map(|m| m.as_str().trim().to_string()),
                alias: cap.get(5).map(|m| m.as_str().trim().to_string()),
                kind,
            })
        })
        .collect()
}

fn pattern() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // Capture groups:
        //   1: optional leading `!` for embeds
        //   2: target name (no `]`, `|`, `#`, `^`)
        //   3: optional `#header`
        //   4: optional `^block-id`
        //   5: optional `|alias`
        Regex::new(r"(!)?\[\[([^\]|#\^]+)(?:#([^\]|\^]+))?(?:\^([^\]|]+))?(?:\|([^\]]+))?\]\]")
            .expect("wiki-link regex must compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(links: &[Wikilink]) -> Vec<&str> {
        links.iter().map(|l| l.target_name.as_str()).collect()
    }

    #[test]
    fn plain_links() {
        let links = extract("see [[Note A]] and [[Note B]]");
        assert_eq!(names(&links), vec!["Note A", "Note B"]);
        assert!(links.iter().all(|l| l.kind == LinkKind::Link));
    }

    #[test]
    fn link_with_alias() {
        let links = extract("[[Real Title|Display]]");
        assert_eq!(links[0].target_name, "Real Title");
        assert_eq!(links[0].alias.as_deref(), Some("Display"));
    }

    #[test]
    fn link_with_header() {
        let links = extract("[[Note#Section]]");
        assert_eq!(links[0].target_name, "Note");
        assert_eq!(links[0].header.as_deref(), Some("Section"));
    }

    #[test]
    fn link_with_block_id() {
        let links = extract("[[Note^abc-123]]");
        assert_eq!(links[0].target_name, "Note");
        assert_eq!(links[0].block_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn link_with_header_and_alias() {
        let links = extract("[[Note#Sect|Display]]");
        assert_eq!(links[0].target_name, "Note");
        assert_eq!(links[0].header.as_deref(), Some("Sect"));
        assert_eq!(links[0].alias.as_deref(), Some("Display"));
    }

    #[test]
    fn embed_marker_distinguishes_link_kind() {
        let links = extract("![[Note]]");
        assert_eq!(links[0].kind, LinkKind::Embed);
    }

    #[test]
    fn ignores_unrelated_brackets() {
        let links = extract("array [[1, 2]] doesn't match either");
        assert_eq!(links[0].target_name, "1, 2");
    }
}
