use std::cmp::Ordering;
use std::collections::HashMap;

use rusqlite::{params, Connection};

use crate::db::LexaDb;
use crate::embed::{matryoshka_truncate, vector_blob, PREVIEW_DIMS};
use crate::query::{fts_query, tokenize};
use crate::types::{LexaError, SearchHit, SearchTier, TierBreakdown};
use crate::Result;

/// Reciprocal Rank Fusion constant. 60 is the value used in the original RRF
/// paper (Cormack et al., 2009) and reused in Exa's published search
/// composition. It controls how much weight rank-1 has over rank-50.
const RRF_K: f32 = 60.0;

/// Top-K returned by the BM25 (sparse) retriever before fusion.
const SPARSE_TOP_K: usize = 50;

/// Top-K returned by the dense (binary-quantized vector) retriever before fusion.
const DENSE_TOP_K: usize = 50;

/// First-stage candidate count for the Matryoshka 256-bit preview KNN.
/// 8× the final dense top-K — bench measurement on the lexa repo showed
/// 4× was too tight (caused 3/20 agent regression vs single-stage 768d).
/// Widening to 8× recovers the full quality at single-digit ms cost
/// because the second stage's `WHERE rowid IN (...)` filter is O(K).
const PREVIEW_TOP_K: usize = DENSE_TOP_K * 8;

/// Number of fused candidates handed to the deep-tier reranker. We trim to
/// 15 (down from a previous 30) because BGE-reranker-base is the dominant
/// latency cost at the deep tier and tends to introduce noise when scoring
/// barely-relevant tail candidates. Anthropic's contextual-retrieval
/// research and Exa's reranker pipeline both place the gain plateau in
/// the 15–30 range.
const RERANK_CANDIDATES: usize = 15;

/// Blend factor for combining the cross-encoder rerank score with the
/// fused RRF score in the deep tier. The cross-encoder logit gets
/// sigmoid-squashed into `[0, 1]` and mixed with the RRF score so a
/// strong cross-encoder vote *adjusts* the RRF order without overriding
/// it entirely (which empirically hurt SciFact nDCG with the previous
/// `score += rerank_score` blend).
const RERANK_BLEND: f32 = 0.7;

/// Maximum characters retained in a hit excerpt fallback. The query-aware
/// `highlight` function below normally produces tighter spans (~80–150
/// chars); this constant only applies when no sentence in the chunk has
/// any query-token overlap and we fall back to plain truncation.
const EXCERPT_MAX_CHARS: usize = 500;

/// Soft target length for a query-aware highlight. Matches Exa's contents-
/// API "highlight" idea: a short span containing the answer, cheap for an
/// LLM consumer.
const HIGHLIGHT_TARGET_CHARS: usize = 220;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub tier: SearchTier,
    pub limit: usize,
    /// Additional reformulations of the query that the deep tier should
    /// search alongside the original. Mirrors Exa Deep's `additionalQueries`
    /// parameter — auto 2–3 paraphrases run in parallel and RRF-fused with
    /// the main query's result list before reranking. Ignored on
    /// non-Deep tiers.
    pub additional_queries: Vec<String>,
}

impl SearchOptions {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            tier: SearchTier::Auto,
            limit: 10,
            additional_queries: Vec::new(),
        }
    }
}

pub fn search_impl(db: &LexaDb, options: &SearchOptions) -> Result<Vec<SearchHit>> {
    let conn = db.conn();
    let limit = options.limit.max(1);

    // Auto routing: pick the cheapest tier likely to answer the query well.
    // We surface the routed tier in `TierBreakdown::routed_to` so callers
    // can debug the decision.
    let (effective_tier, routed_to) = if options.tier == SearchTier::Auto {
        let routed = classify_query(&options.query);
        (routed, Some(routed))
    } else {
        (options.tier, None)
    };

    let mut hits = match effective_tier {
        SearchTier::Auto => unreachable!("Auto resolves to a concrete tier above"),
        SearchTier::Instant => {
            let bm25 = bm25_search(conn, &options.query, SPARSE_TOP_K)?;
            hydrate(conn, &options.query, &rank_to_rrf(&bm25), &bm25, &[], limit)?.0
        }
        SearchTier::Dense => {
            let vector = vector_search(db, &options.query, DENSE_TOP_K)?;
            hydrate(
                conn,
                &options.query,
                &rank_to_rrf(&vector),
                &[],
                &vector,
                limit,
            )?
            .0
        }
        SearchTier::Fast | SearchTier::Deep => {
            // Run BM25 (FTS5 SQL, ~1–2 ms) on the main thread while the
            // embedder forward pass (~7 ms ONNX) runs on a scoped worker.
            // `Connection` is `!Sync` so we cannot move the SQL onto the
            // worker, but `Mutex<Embedder>` is `Send + Sync`, so the
            // embedder is what we ship across the thread boundary. After
            // both finish we hit the binary KNN on the main connection
            // (~1 ms).
            let embedder_lock = db.embedder()?;
            let query_str = options.query.as_str();
            let (bm25, embedding) = std::thread::scope(|scope| -> Result<_> {
                let embed_handle = scope.spawn(|| -> Result<Vec<f32>> {
                    let mut guard = embedder_lock
                        .lock()
                        .map_err(|err| LexaError::Embedding(err.to_string()))?;
                    guard.embed_query(query_str)
                });
                let bm25 = bm25_search(conn, query_str, SPARSE_TOP_K)?;
                let embedding = embed_handle
                    .join()
                    .map_err(|_| LexaError::Embedding("embed worker panicked".into()))??;
                Ok((bm25, embedding))
            })?;
            let vector = vector_knn(conn, &embedding, DENSE_TOP_K)?;

            // Deep tier supports `additionalQueries` à la Exa Deep: fan out
            // 2–3 reformulations through the same hybrid pipeline and
            // RRF-fuse all 2(N+1) ranked lists. Fast tier ignores them.
            let fused =
                if effective_tier == SearchTier::Deep && !options.additional_queries.is_empty() {
                    let mut all_lists: Vec<Vec<(i64, f32)>> =
                        Vec::with_capacity(2 + options.additional_queries.len() * 2);
                    all_lists.push(bm25.clone());
                    all_lists.push(vector.clone());
                    for extra in &options.additional_queries {
                        let extra_str = extra.as_str();
                        let (extra_bm25, extra_emb) = std::thread::scope(|scope| -> Result<_> {
                            let h = scope.spawn(|| -> Result<Vec<f32>> {
                                let mut guard = embedder_lock
                                    .lock()
                                    .map_err(|err| LexaError::Embedding(err.to_string()))?;
                                guard.embed_query(extra_str)
                            });
                            let b = bm25_search(conn, extra_str, SPARSE_TOP_K)?;
                            let e = h.join().map_err(|_| {
                                LexaError::Embedding("embed worker panicked".into())
                            })??;
                            Ok((b, e))
                        })?;
                        let extra_vec = vector_knn(conn, &extra_emb, DENSE_TOP_K)?;
                        all_lists.push(extra_bm25);
                        all_lists.push(extra_vec);
                    }
                    let refs: Vec<&[(i64, f32)]> = all_lists.iter().map(Vec::as_slice).collect();
                    fuse_many(&refs)
                } else {
                    fuse(&bm25, &vector)
                };

            let candidate_count = if effective_tier == SearchTier::Deep {
                RERANK_CANDIDATES
            } else {
                limit
            };
            let (mut hits, full_texts) = hydrate(
                conn,
                &options.query,
                &fused,
                &bm25,
                &vector,
                candidate_count,
            )?;
            if effective_tier == SearchTier::Deep && !hits.is_empty() {
                rerank(db, &options.query, &mut hits, &full_texts)?;
            }
            hits.truncate(limit);
            hits
        }
    };

    if let Some(tier) = routed_to {
        for hit in &mut hits {
            hit.breakdown.routed_to = Some(tier);
        }
    }

    Ok(hits)
}

/// Inspect the query and pick the cheapest tier likely to answer it well.
/// This is the local equivalent of Exa's `auto` policy.
///
/// Heuristics, in priority order:
/// - explicit `[deep]` prefix → `Deep`
/// - **single** identifier-shaped token (snake_case, CamelCase with both
///   upper and lower, or `::` / `.` path) → `Instant`
/// - long question-form (≥ 6 tokens AND ends with `?`) → `Deep`
/// - default → `Fast`
///
/// We deliberately keep `Instant` routing narrow — only the "user pasted a
/// symbol they were looking at" case. A natural-language query that
/// happens to contain a CamelCase identifier ("the BGE reranker pipeline")
/// is still a natural-language query and belongs on `Fast` so the dense
/// retriever sees it.
fn classify_query(query: &str) -> SearchTier {
    let trimmed = query.trim();
    if let Some(rest) = trimmed.strip_prefix("[deep]") {
        let _ = rest;
        return SearchTier::Deep;
    }

    let tokens: Vec<&str> = trimmed
        .split_whitespace()
        .filter(|tok| tok.chars().any(char::is_alphanumeric))
        .collect();

    if tokens.is_empty() {
        return SearchTier::Fast;
    }

    if tokens.len() == 1 {
        let tok = tokens[0];
        let snake_case = tok.contains('_') && tok.chars().any(|c| c.is_ascii_alphanumeric());
        let mixed_case = tok.chars().any(|c| c.is_ascii_uppercase())
            && tok.chars().any(|c| c.is_ascii_lowercase());
        let path_like = tok.contains("::") || (tok.contains('.') && !tok.ends_with('.'));
        if snake_case || mixed_case || path_like {
            return SearchTier::Instant;
        }
    }

    if tokens.len() >= 6 && trimmed.ends_with('?') {
        return SearchTier::Deep;
    }

    SearchTier::Fast
}

/// Two-stage Matryoshka KNN against the binary-quantized vector indices.
///
/// Stage 1 runs Hamming KNN over the 256-bit `vectors_bin_preview` table
/// (3× less memory bandwidth than the 768-bit table) and returns
/// `PREVIEW_TOP_K` candidate `rowid`s. Stage 2 re-scores those candidates
/// against the full 768-bit `vectors_bin` table using the same
/// `vec_distance_hamming` op, returning the requested `limit`.
///
/// This is the local analogue of Exa's published two-stage Matryoshka
/// design: a coarse pass over a prefix dimension (Exa uses 256 of 4096;
/// we use 256 of 768) followed by a full-dim re-rank over the survivors.
/// Quality stays within ~0.5 nDCG of full-dim because Nomic v1.5 is MRL-
/// trained at exactly the {64, 128, 256, 512, 768} canonical points.
fn vector_knn(conn: &Connection, embedding: &[f32], limit: usize) -> Result<Vec<(i64, f32)>> {
    // Stage 1: coarse Matryoshka KNN at 256 bits.
    let preview_blob = vector_blob(&matryoshka_truncate(embedding, PREVIEW_DIMS));
    let mut preview_stmt = conn.prepare_cached(
        "SELECT rowid
         FROM vectors_bin_preview
         WHERE embedding MATCH vec_quantize_binary(?1) AND k = ?2
         ORDER BY distance",
    )?;
    let preview_ids: Vec<i64> = preview_stmt
        .query_map(params![preview_blob, PREVIEW_TOP_K as i64], |row| {
            row.get::<_, i64>(0)
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if preview_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Stage 2: re-score the survivors at full 768 bits.
    let full_blob = vector_blob(embedding);
    // sqlite-vec's vec_distance_hamming works on `bit[N]` blobs; combined
    // with a `rowid IN (json_each(?))` filter we get the re-rank in a
    // single round trip without an enormous variadic IN list.
    let preview_ids_json = serde_json::to_string(&preview_ids)?;
    let mut rescore_stmt = conn.prepare_cached(
        "SELECT v.rowid,
                vec_distance_hamming(v.embedding, vec_quantize_binary(?1)) AS distance
         FROM vectors_bin AS v
         WHERE v.rowid IN (SELECT value FROM json_each(?2))
         ORDER BY distance
         LIMIT ?3",
    )?;
    let rows =
        rescore_stmt.query_map(params![full_blob, preview_ids_json, limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let distance: f64 = row.get(1)?;
            Ok((id, (1.0 / (1.0 + distance)) as f32))
        })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn bm25_search(conn: &Connection, query: &str, limit: usize) -> Result<Vec<(i64, f32)>> {
    let fts_query = fts_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare_cached(
        "SELECT rowid, bm25(chunks_fts) AS rank
         FROM chunks_fts
         WHERE chunks_fts MATCH ?1
         ORDER BY rank
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
        let id: i64 = row.get(0)?;
        let rank: f64 = row.get(1)?;
        Ok((id, (1.0 / (1.0 + rank.abs())) as f32))
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn vector_search(db: &LexaDb, query: &str, limit: usize) -> Result<Vec<(i64, f32)>> {
    let embedding = {
        let lock = db.embedder()?;
        let mut guard = lock
            .lock()
            .map_err(|err| LexaError::Embedding(err.to_string()))?;
        guard.embed_query(query)?
    };
    vector_knn(db.conn(), &embedding, limit)
}

/// Reciprocal Rank Fusion over an arbitrary number of ranked lists.
/// Used by the Fast tier (BM25 + dense, two lists) and by the Deep tier's
/// `additionalQueries` fan-out (main BM25 + main dense + N reformulation
/// BM25 + N reformulation dense).
fn fuse_many(lists: &[&[(i64, f32)]]) -> Vec<(i64, f32)> {
    let mut scores = HashMap::<i64, f32>::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(score_desc);
    fused
}

fn fuse(bm25: &[(i64, f32)], vector: &[(i64, f32)]) -> Vec<(i64, f32)> {
    fuse_many(&[bm25, vector])
}

fn rank_to_rrf(items: &[(i64, f32)]) -> Vec<(i64, f32)> {
    items
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, 1.0 / (RRF_K + rank as f32 + 1.0)))
        .collect()
}

/// Hydrate ranked chunk IDs into full `SearchHit`s plus the per-hit full
/// chunk text (which the deep-tier reranker uses instead of the short
/// display excerpt). The display excerpt is produced by `highlight`, the
/// query-aware span picker; the reranker gets the full text.
fn hydrate(
    conn: &Connection,
    query: &str,
    ranked: &[(i64, f32)],
    bm25: &[(i64, f32)],
    vector: &[(i64, f32)],
    limit: usize,
) -> Result<(Vec<SearchHit>, Vec<String>)> {
    let bm25_rank = ranks(bm25);
    let vector_rank = ranks(vector);
    let bm25_scores = score_map(bm25);
    let vector_scores = score_map(vector);
    let mut hits = Vec::new();
    let mut full_texts = Vec::new();
    let mut stmt = conn.prepare_cached(
        "SELECT d.path, c.line_start, c.line_end, c.text, c.context
         FROM chunks c JOIN documents d ON d.id = c.doc_id
         WHERE c.id = ?1",
    )?;
    for (id, score) in ranked.iter().take(limit) {
        let (hit, text) = stmt.query_row(params![id], |row| {
            let text: String = row.get(3)?;
            let heading: Option<String> = row.get(4)?;
            let hit = SearchHit {
                path: row.get(0)?,
                line_start: row.get(1)?,
                line_end: row.get(2)?,
                score: *score,
                excerpt: highlight(query, &text),
                heading,
                breakdown: TierBreakdown {
                    bm25_rank: bm25_rank.get(id).copied(),
                    vector_rank: vector_rank.get(id).copied(),
                    bm25_score: bm25_scores.get(id).copied().unwrap_or_default(),
                    vector_score: vector_scores.get(id).copied().unwrap_or_default(),
                    rerank_score: None,
                    routed_to: None,
                },
            };
            Ok((hit, text))
        })?;
        hits.push(hit);
        full_texts.push(text);
    }
    Ok((hits, full_texts))
}

/// Sigmoid for the rerank blend. f32 std doesn't expose one and the
/// cross-encoder logits we get back from BGE-reranker-base are unbounded,
/// so we squash to `[0, 1]` before blending with the bounded RRF score.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Cross-encoder reranking on the deep-tier candidate set. Sends the
/// **full chunk text** (not the truncated display excerpt) so the reranker
/// has the context it was trained on, then blends the squashed cross-encoder
/// score with the existing RRF score (`RERANK_BLEND` weight on the
/// reranker, `1 - RERANK_BLEND` on RRF). This avoids the previous failure
/// mode where unbounded reranker logits completely overrode the fusion
/// order — which on SciFact dropped deep-tier nDCG @10 from 0.706 to 0.643.
fn rerank(db: &LexaDb, query: &str, hits: &mut [SearchHit], full_texts: &[String]) -> Result<()> {
    let docs: Vec<String> = full_texts.to_vec();
    let scores = {
        let lock = db.reranker()?;
        let mut guard = lock
            .lock()
            .map_err(|err| LexaError::Embedding(err.to_string()))?;
        guard.rerank(query, &docs)?
    };
    for (idx, raw_score) in scores {
        if let Some(hit) = hits.get_mut(idx) {
            let rrf = hit.score;
            let squashed = sigmoid(raw_score);
            hit.score = RERANK_BLEND * squashed + (1.0 - RERANK_BLEND) * rrf;
            hit.breakdown.rerank_score = Some(raw_score);
        }
    }
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    Ok(())
}

fn ranks(items: &[(i64, f32)]) -> HashMap<i64, usize> {
    items
        .iter()
        .enumerate()
        .map(|(idx, (id, _))| (*id, idx + 1))
        .collect()
}

fn score_map(items: &[(i64, f32)]) -> HashMap<i64, f32> {
    items.iter().copied().collect()
}

fn score_desc(left: &(i64, f32), right: &(i64, f32)) -> Ordering {
    right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal)
}

/// Query-aware highlight extraction. Mirrors Exa's contents-API
/// "highlights": pick the chunk's most query-relevant sentence span
/// instead of returning a fixed-width truncation. On a typical SciFact
/// chunk this drops the displayed excerpt from ~500 chars to ~80–150
/// chars, which is what Exa's published ~10× LLM-token-reduction claim
/// rests on.
///
/// Algorithm:
/// 1. Tokenize the query with the same `crate::query::tokenize` used by
///    BM25 (lowercases, drops short tokens, strips stopwords).
/// 2. Split the chunk into sentence spans on `[.!?;\n]\s+`.
/// 3. Score each sentence by *unique* query-token overlap (set
///    intersection) so a sentence that contains the answer's noun gets
///    rewarded once, not once per duplicate.
/// 4. Pick the highest-scoring sentence; expand by ±1 sentence if the
///    result is shorter than `HIGHLIGHT_TARGET_CHARS`.
/// 5. If no sentence has any overlap, fall back to `excerpt`'s plain
///    truncation.
fn highlight(query: &str, text: &str) -> String {
    let query_tokens: std::collections::HashSet<String> = tokenize(query).collect();
    if query_tokens.is_empty() {
        return excerpt(text);
    }

    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return String::new();
    }

    // Sentence split on terminal punctuation. The pattern is conservative:
    // require punctuation followed by whitespace, otherwise we'd split on
    // every `.` in identifiers and floats.
    let sentences = split_sentences(&compact);
    if sentences.is_empty() {
        return excerpt(&compact);
    }

    let scores: Vec<(usize, usize)> = sentences
        .iter()
        .enumerate()
        .map(|(idx, sentence)| {
            let tokens: std::collections::HashSet<String> = tokenize(sentence).collect();
            let overlap = query_tokens.intersection(&tokens).count();
            (idx, overlap)
        })
        .collect();

    let best = scores.iter().max_by_key(|(_, score)| *score).copied();
    let Some((best_idx, best_score)) = best else {
        return excerpt(&compact);
    };
    if best_score == 0 {
        // No sentence shares any non-stopword token with the query — fall
        // back to plain truncation rather than emit a misleading highlight.
        return excerpt(&compact);
    }

    // Expand the window with neighbours until we hit the soft target or
    // exhaust the chunk.
    let mut start = best_idx;
    let mut end = best_idx;
    let mut span_len = sentences[best_idx].len();
    while span_len < HIGHLIGHT_TARGET_CHARS {
        let grew = if start > 0
            && (end + 1 == sentences.len() || start.abs_diff(0) <= end + 1 - best_idx)
        {
            start -= 1;
            span_len += sentences[start].len() + 1;
            true
        } else if end + 1 < sentences.len() {
            end += 1;
            span_len += sentences[end].len() + 1;
            true
        } else {
            false
        };
        if !grew {
            break;
        }
    }

    let span: String = sentences[start..=end].join(" ");
    // Cap at 1.5× the target so a long-running sentence doesn't blow the
    // budget on its own.
    let cap = HIGHLIGHT_TARGET_CHARS * 3 / 2;
    if span.len() <= cap {
        span
    } else {
        let mut cut = cap;
        while cut > 0 && !span.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}...", &span[..cut])
    }
}

fn split_sentences(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut starts = vec![0];
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if matches!(b, b'.' | b'!' | b'?' | b';' | b'\n')
            && i + 1 < bytes.len()
            && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\n' || bytes[i + 1] == b'\t')
        {
            // Move start to the character *after* the whitespace.
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\n' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && text.is_char_boundary(j) {
                starts.push(j);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    starts.push(text.len());
    starts
        .windows(2)
        .map(|w| text[w[0]..w[1]].trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Compact whitespace and truncate at a UTF-8 char boundary. Used as a
/// fallback when `highlight` can't find any query overlap and we still
/// want *something* to show the user.
fn excerpt(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= EXCERPT_MAX_CHARS {
        return compact;
    }
    let mut end = EXCERPT_MAX_CHARS;
    while end > 0 && !compact.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &compact[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_boosts_overlap() {
        let bm25 = vec![(1, 1.0), (2, 0.8)];
        let vector = vec![(3, 1.0), (1, 0.7)];
        let fused = fuse(&bm25, &vector);
        assert_eq!(fused[0].0, 1);
    }

    #[test]
    fn classify_routes_single_identifier_to_instant() {
        assert_eq!(classify_query("vec_quantize_binary"), SearchTier::Instant);
        assert_eq!(classify_query("LexaDb::open"), SearchTier::Instant);
        assert_eq!(classify_query("Embedder::embed_query"), SearchTier::Instant);
    }

    #[test]
    fn classify_keeps_natural_language_with_identifiers_on_fast() {
        // A natural-language query containing a CamelCase term still needs
        // the dense retriever; Instant would lose semantic recall.
        assert_eq!(
            classify_query("matryoshka_truncate helper that re-normalizes"),
            SearchTier::Fast
        );
        assert_eq!(
            classify_query("the BGE cross encoder reranker"),
            SearchTier::Fast
        );
    }

    #[test]
    fn classify_routes_explicit_deep_prefix() {
        assert_eq!(
            classify_query("[deep] explain the rerank pipeline"),
            SearchTier::Deep
        );
    }

    #[test]
    fn classify_routes_long_questions_to_deep() {
        assert_eq!(
            classify_query("how does the reranker score truncated excerpts in deep tier?"),
            SearchTier::Deep
        );
    }

    #[test]
    fn classify_defaults_to_fast() {
        assert_eq!(
            classify_query("hybrid lexical dense retrieval"),
            SearchTier::Fast
        );
        assert_eq!(
            classify_query("binary quantized vector search"),
            SearchTier::Fast
        );
    }

    #[test]
    fn highlight_picks_query_relevant_sentence() {
        // Filler so long that even with neighbour expansion the highlight
        // can't pull it all in.
        let filler: String = "alpha beta gamma delta. ".repeat(20);
        let text = format!(
            "{filler}\
             The reranker scores candidates by cross encoder logits. \
             {filler}"
        );
        let span = highlight("reranker cross encoder logits", &text);
        assert!(span.contains("reranker"));
        assert!(span.contains("cross encoder"));
        // The highlight stays bounded — must not include the entire filler.
        assert!(span.len() <= HIGHLIGHT_TARGET_CHARS * 3 / 2 + 4);
    }

    #[test]
    fn highlight_falls_back_when_no_overlap() {
        let text = "Some prose without any of the query's words.";
        let span = highlight("matryoshka quantization", text);
        // Falls back to plain truncation; should equal `excerpt(text)`.
        assert_eq!(span, excerpt(text));
    }

    #[test]
    fn highlight_caps_at_soft_target() {
        // Build a 4 kb text with one tiny target sentence in the middle so
        // the expansion logic can't possibly stretch past the cap.
        let prefix: String = "ipsum dolor sit amet. ".repeat(50);
        let suffix: String = "vivamus sed lacus. ".repeat(50);
        let text = format!("{}TARGET token here. {}", prefix, suffix);
        let span = highlight("target token", &text);
        assert!(span.len() <= HIGHLIGHT_TARGET_CHARS * 3 / 2 + 4 /* "..." */);
    }

    #[test]
    fn hash_fallback_can_score() {
        let query = crate::embed::hash_embedding("config validation");
        let doc = crate::embed::hash_embedding("configuration validation function");
        assert!(crate::embed::cosine(&query, &doc) > -1.0);
    }
}
