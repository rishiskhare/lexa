//! Obsidian tag extraction.
//!
//! Tags come from two places:
//!
//! 1. Frontmatter `tags:` (string, comma-list, or YAML sequence). The
//!    `frontmatter` module already returns those as `Frontmatter::tags`.
//! 2. Inline `#tag` syntax in the body, e.g. `#project/lexa`. Inline
//!    tags must be skipped inside fenced code blocks (`` ``` ``) and on
//!    heading lines (a leading `#` followed by a space — that's a
//!    heading, not a tag).
//!
//! Result is lowercase-normalised so `#Project` and `#project` collapse,
//! and de-duplicated within a single note.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::frontmatter::Frontmatter;

/// Extract every distinct, lowercase-normalised tag for a note.
///
/// `body` is the markdown body (frontmatter already stripped); `fm`
/// supplies the frontmatter-declared tags.
pub fn extract(body: &str, fm: &Frontmatter) -> Vec<String> {
    let mut out = HashSet::<String>::new();

    for tag in &fm.tags {
        let normalised = tag.trim_start_matches('#').trim().to_ascii_lowercase();
        if !normalised.is_empty() {
            out.insert(normalised);
        }
    }

    let inline = inline_pattern();
    let mut in_fence = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if is_heading(trimmed) {
            continue;
        }
        for cap in inline.captures_iter(line) {
            if let Some(m) = cap.get(1) {
                let tag = m.as_str().to_ascii_lowercase();
                if !tag.is_empty() {
                    out.insert(tag);
                }
            }
        }
    }

    let mut tags: Vec<String> = out.into_iter().collect();
    tags.sort();
    tags
}

/// `#tag` (or `#nested/tag`) preceded by start-of-line or whitespace.
fn inline_pattern() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?:^|\s)#([A-Za-z][A-Za-z0-9_/\-]*)").expect("inline tag regex must compile")
    })
}

/// A markdown heading is `#` ... `######` followed by a space.
fn is_heading(trimmed_line: &str) -> bool {
    let mut chars = trimmed_line.chars();
    let mut hashes = 0;
    for c in chars.by_ref() {
        if c == '#' {
            hashes += 1;
            if hashes > 6 {
                return false;
            }
        } else {
            return hashes >= 1 && c == ' ';
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fm_with_tags(tags: Vec<&str>) -> Frontmatter {
        Frontmatter {
            tags: tags.into_iter().map(String::from).collect(),
            ..Frontmatter::default()
        }
    }

    #[test]
    fn merges_frontmatter_and_inline() {
        let fm = fm_with_tags(vec!["project", "research"]);
        let body = "Working on #project today, also #fitness.\n";
        let tags = extract(body, &fm);
        assert_eq!(tags, vec!["fitness", "project", "research"]);
    }

    #[test]
    fn lowercases_and_dedupes() {
        let fm = fm_with_tags(vec!["Project", "PROJECT"]);
        let body = "#Project #project\n";
        let tags = extract(body, &fm);
        assert_eq!(tags, vec!["project"]);
    }

    #[test]
    fn skips_code_fences() {
        let fm = Frontmatter::default();
        let body = "before\n```\n#nottag\n```\nafter #realtag\n";
        let tags = extract(body, &fm);
        assert_eq!(tags, vec!["realtag"]);
    }

    #[test]
    fn skips_heading_lines() {
        let fm = Frontmatter::default();
        let body = "## Heading text\nbody #realtag\n";
        let tags = extract(body, &fm);
        assert_eq!(tags, vec!["realtag"]);
    }

    #[test]
    fn supports_nested_tags() {
        let fm = Frontmatter::default();
        let body = "#project/lexa is a sub-tag.\n";
        let tags = extract(body, &fm);
        assert_eq!(tags, vec!["project/lexa"]);
    }

    #[test]
    fn ignores_url_fragments_and_html_ids() {
        let fm = Frontmatter::default();
        let body = "see http://example.com#section or <div id=\"foo\">\n";
        let tags = extract(body, &fm);
        assert!(tags.is_empty());
    }
}
