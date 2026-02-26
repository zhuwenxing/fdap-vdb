use vdb_common::error::Result;

/// A search result: (internal row id, distance).
pub type SearchResult = (u64, f32);

/// Trait for vector index implementations.
pub trait VectorIndex: Send + Sync {
    /// Insert vectors with their associated row IDs.
    fn insert(&self, ids: &[u64], vectors: &[&[f32]]) -> Result<()>;

    /// Search for the nearest `k` neighbors of `query`.
    /// `ef` controls the search quality for HNSW (ignored by flat index).
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>>;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the index to bytes.
    fn save(&self) -> Result<Vec<u8>>;

    /// Dimension of vectors in this index.
    fn dimension(&self) -> usize;
}
