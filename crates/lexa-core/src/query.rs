//! Query analysis: tokenization, stopword pruning, and FTS5 query construction.
//!
//! The `fts_query` builder follows the same philosophy described in Exa's BM25
//! optimization writeup: rely on BM25's IDF component (which SQLite FTS5 ships
//! natively) to weight terms by rarity, and let `OR` fusion match documents
//! that share even a subset of query terms. A small high-confidence stopword
//! list strips the most-common English fillers up front so they don't dominate
//! posting-list merges on large corpora.

use std::collections::HashSet;

/// Build an FTS5 MATCH expression from a free-form query.
///
/// Tokens are lowercased, deduplicated, stopword-filtered, escaped, and joined
/// with `OR`. Returns an empty string when the query has no surviving tokens
/// so callers can short-circuit the FTS lookup.
pub(crate) fn fts_query(query: &str) -> String {
    let mut seen = HashSet::with_capacity(query.len() / 4);
    let tokens: Vec<String> = tokenize(query)
        .filter(|token| !is_stopword(token))
        .filter(|token| seen.insert(token.clone()))
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect();
    tokens.join(" OR ")
}

/// Lowercase ASCII-alphanumeric tokenizer. Drops tokens shorter than 2 chars.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|raw| {
            let token = raw.trim().to_ascii_lowercase();
            (token.len() > 1).then_some(token)
        })
}

/// Returns `true` for the most-frequent English fillers that BM25 IDF still
/// fails to fully suppress on small corpora. The slice is `const`-sorted and
/// queried via binary search; no allocations, no lazy init, no HashSet.
///
/// The list is intentionally short and conservative — narrowed to the words
/// that actually drift to the top of FTS5 posting lists on natural-language
/// queries. Bigger published lists (NLTK's 179, SMART's 571) include words
/// like "computer" and "describe" that carry genuine information for code
/// search. We keep only the function words.
fn is_stopword(token: &str) -> bool {
    STOPWORDS.binary_search(&token).is_ok()
}

/// **MUST stay sorted (ASCII order).** A `debug_assert!` in tests guards this.
const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "might",
    "must",
    "my",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopword_list_is_sorted_and_unique() {
        for window in STOPWORDS.windows(2) {
            assert!(
                window[0] < window[1],
                "STOPWORDS must be sorted and unique; offender: {:?}",
                window
            );
        }
    }

    #[test]
    fn fts_query_quotes_terms_and_uses_or() {
        assert_eq!(
            fts_query("config validation function"),
            "\"config\" OR \"validation\" OR \"function\""
        );
    }

    #[test]
    fn fts_query_drops_stopwords() {
        assert_eq!(
            fts_query("what is the role of ascorbate"),
            "\"role\" OR \"ascorbate\""
        );
    }

    #[test]
    fn fts_query_handles_quotes_and_punctuation() {
        // Surviving alphanumeric tokens get escaped quotes, punctuation breaks tokens.
        let q = fts_query(r#"foo "bar" baz!quux"#);
        assert_eq!(q, "\"foo\" OR \"bar\" OR \"baz\" OR \"quux\"");
    }

    #[test]
    fn fts_query_empty_for_all_stopwords() {
        assert!(fts_query("the and of").is_empty());
    }

    #[test]
    fn tokenize_lowercases_and_dedupes_via_caller() {
        let toks: Vec<_> = tokenize("Foo bar BAR baz").collect();
        assert_eq!(toks, vec!["foo", "bar", "bar", "baz"]);
    }

    #[test]
    fn is_stopword_matches_via_binary_search() {
        assert!(is_stopword("the"));
        assert!(is_stopword("which"));
        assert!(!is_stopword("ascorbate"));
        assert!(!is_stopword("config"));
    }
}
