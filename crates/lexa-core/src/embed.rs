//! Embedding + reranking inference.
//!
//! Production backbone is **`nomic-ai/nomic-embed-text-v1.5` (quantized)**:
//! 768-dim, Matryoshka-trained, Apache-2 licensed. The deep-tier reranker is
//! `BAAI/bge-reranker-base`. The `Hash` backend is a deterministic
//! FNV-1a-into-fixed-dim fallback used only by tests and CI smoke runs.

use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};

use crate::{LexaError, Result};

/// Native embedding dimension. Tied to the model below; both must change
/// together (and the existing `vectors_bin` schema must be re-indexed).
pub const EMBEDDING_DIMS: usize = 768;

/// Matryoshka preview dimension. Nomic v1.5 was MRL-trained at canonical
/// prefix dims {64, 128, 256, 512, 768}; 256 is the published Exa choice
/// for the coarse retrieval pass. We store a second binary-quantized index
/// at this width and use it as the first stage of two-stage KNN — coarse
/// 256-bit Hamming over the whole corpus, then full 768-bit Hamming over
/// the top-K survivors.
pub const PREVIEW_DIMS: usize = 256;

/// Task prefix prepended to *queries* before embedding. Nomic v1.5 was
/// trained with this asymmetric pair; using a query without the prefix
/// silently drops nDCG by several points.
const QUERY_PREFIX: &str = "search_query: ";

/// Task prefix prepended to *documents* before embedding.
const DOCUMENT_PREFIX: &str = "search_document: ";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EmbeddingBackend {
    /// Real ONNX inference via fastembed-rs.
    FastEmbed,
    /// Deterministic FNV-1a hashing into a fixed-dim vector. CI / offline only.
    Hash,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub backend: EmbeddingBackend,
    pub show_download_progress: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let backend = match std::env::var("LEXA_EMBEDDER").ok().as_deref() {
            Some("hash") => EmbeddingBackend::Hash,
            _ => EmbeddingBackend::FastEmbed,
        };
        Self {
            backend,
            show_download_progress: true,
        }
    }
}

pub enum Embedder {
    Fast(Box<TextEmbedding>),
    Hash,
}

impl Embedder {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        match config.backend {
            EmbeddingBackend::Hash => Ok(Self::Hash),
            EmbeddingBackend::FastEmbed => {
                let options = InitOptions::new(EmbeddingModel::NomicEmbedTextV15Q)
                    .with_show_download_progress(config.show_download_progress);
                TextEmbedding::try_new(options)
                    .map(Box::new)
                    .map(Self::Fast)
                    .map_err(|error| LexaError::Embedding(error.to_string()))
            }
        }
    }

    /// Encode a batch of *documents* (passages to be indexed).
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = match self {
            Self::Fast(_) => texts
                .iter()
                .map(|text| format!("{DOCUMENT_PREFIX}{text}"))
                .collect(),
            Self::Hash => texts.to_vec(),
        };
        self.encode(&prefixed)
    }

    /// Encode a *query* string. Symmetric with `embed_documents` — without
    /// the matching task prefix, retrieval quality drops measurably.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prefixed = match self {
            Self::Fast(_) => format!("{QUERY_PREFIX}{query}"),
            Self::Hash => query.to_string(),
        };
        Ok(self.encode(&[prefixed])?.remove(0))
    }

    fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Fast(model) => model
                .embed(texts, None)
                .map_err(|error| LexaError::Embedding(error.to_string())),
            Self::Hash => Ok(texts.iter().map(|text| hash_embedding(text)).collect()),
        }
    }
}

pub enum Reranker {
    Fast(Box<TextRerank>),
    Hash,
}

impl Reranker {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        match config.backend {
            EmbeddingBackend::Hash => Ok(Self::Hash),
            EmbeddingBackend::FastEmbed => {
                let options = RerankInitOptions::new(RerankerModel::BGERerankerBase)
                    .with_show_download_progress(config.show_download_progress);
                TextRerank::try_new(options)
                    .map(Box::new)
                    .map(Self::Fast)
                    .map_err(|error| LexaError::Embedding(error.to_string()))
            }
        }
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>> {
        match self {
            Self::Fast(model) => {
                let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
                model
                    .rerank(query, refs, false, None)
                    .map(|items| {
                        items
                            .into_iter()
                            .map(|item| (item.index, item.score))
                            .collect()
                    })
                    .map_err(|error| LexaError::Embedding(error.to_string()))
            }
            Self::Hash => {
                let q = hash_embedding(query);
                let mut scores: Vec<(usize, f32)> = documents
                    .iter()
                    .enumerate()
                    .map(|(idx, text)| (idx, cosine(&q, &hash_embedding(text))))
                    .collect();
                scores.sort_by(|left, right| {
                    right
                        .1
                        .partial_cmp(&left.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(scores)
            }
        }
    }
}

/// Truncate a Matryoshka-trained embedding to a smaller prefix and
/// re-normalize. Nomic v1.5 is MRL-trained at canonical dims 64, 128, 256,
/// 512, 768, so any prefix is a valid embedding in the same vector space.
/// fastembed already returns L2-normalized embeddings; we re-normalize after
/// truncation so cosine scores stay in `[-1, 1]`.
pub fn matryoshka_truncate(vector: &[f32], target_dims: usize) -> Vec<f32> {
    let take = target_dims.min(vector.len());
    let mut out = vector[..take].to_vec();
    let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut out {
            *value /= norm;
        }
    }
    out
}

pub fn hash_embedding(text: &str) -> Vec<f32> {
    let mut out = vec![0.0; EMBEDDING_DIMS];
    for token in tokenize(text) {
        let hash = fnv1a(token.as_bytes());
        let idx = (hash as usize) % EMBEDDING_DIMS;
        let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
        out[idx] += sign;
    }
    normalize(&mut out);
    out
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|raw| {
            let token = raw.trim().to_ascii_lowercase();
            (token.len() > 1).then_some(token)
        })
        .collect()
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}

/// Pack an f32 embedding into a raw little-endian byte buffer.
///
/// `sqlite-vec` accepts both JSON arrays and raw f32 BLOBs of length
/// `dims * 4` bytes; the BLOB form skips the JSON tokenizer on every insert
/// and every query. Both x86_64 and arm64 are little-endian, so `to_ne_bytes`
/// matches what sqlite-vec's `memcpy` reader expects.
pub fn vector_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matryoshka_truncate_normalizes() {
        let v = vec![3.0, 4.0, 0.0, 0.0];
        let t = matryoshka_truncate(&v, 2);
        assert_eq!(t.len(), 2);
        let norm = t.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((t[0] - 0.6).abs() < 1e-6);
        assert!((t[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn matryoshka_truncate_caps_at_input_len() {
        let v = vec![1.0, 0.0, 0.0];
        assert_eq!(matryoshka_truncate(&v, 8).len(), 3);
    }

    #[test]
    fn hash_embedding_has_canonical_dims() {
        assert_eq!(hash_embedding("hello world").len(), EMBEDDING_DIMS);
    }
}
