use std::path::Path;

use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use vdb_common::config::IndexConfig;
use vdb_common::error::{Result, VdbError};
use vdb_common::metrics::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

/// HNSW-based vector index wrapping `hnsw_rs`.
///
/// For cosine metric, vectors are normalized before insertion and search,
/// then DistDot is used (for normalized vectors, dot product = cosine similarity).
pub struct HnswIndex {
    dimension: usize,
    metric: DistanceMetric,
    ef_search: usize,
    // We use Box to create a self-referential structure:
    // HnswIo must be kept alive as long as Hnsw exists (for mmap).
    // We use 'static by leaking HnswIo for loaded indexes.
    inner: RwLock<HnswInner>,
    // Keep the leaked HnswIo alive
    _io: Option<Box<HnswIo>>,
}

// Safety: HnswIo and Hnsw are Send + Sync through their internal locking
unsafe impl Send for HnswIndex {}
unsafe impl Sync for HnswIndex {}

enum HnswInner {
    L2(Hnsw<'static, f32, DistL2>),
    Dot(Hnsw<'static, f32, DistDot>),
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

impl HnswIndex {
    pub fn new(dimension: usize, metric: DistanceMetric, config: &IndexConfig) -> Self {
        let inner = match metric {
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => HnswInner::Dot(Hnsw::new(
                config.max_nb_connection,
                config.max_elements,
                16,
                config.ef_construction,
                DistDot {},
            )),
            DistanceMetric::L2 => HnswInner::L2(Hnsw::new(
                config.max_nb_connection,
                config.max_elements,
                16,
                config.ef_construction,
                DistL2 {},
            )),
        };
        Self {
            dimension,
            metric,
            ef_search: config.ef_search,
            inner: RwLock::new(inner),
            _io: None,
        }
    }

    /// Save the HNSW index to a directory.
    pub fn save_to_dir(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        match &*self.inner.read() {
            HnswInner::L2(hnsw) => {
                hnsw.file_dump(dir, "hnsw")
                    .map_err(|e| VdbError::Index(format!("failed to save HNSW: {e}")))?;
            }
            HnswInner::Dot(hnsw) => {
                hnsw.file_dump(dir, "hnsw")
                    .map_err(|e| VdbError::Index(format!("failed to save HNSW: {e}")))?;
            }
        }
        Ok(())
    }

    /// Load HNSW index from a directory.
    pub fn load_from_dir(
        dimension: usize,
        metric: DistanceMetric,
        config: &IndexConfig,
        dir: &Path,
    ) -> Result<Self> {
        // Leak HnswIo to get 'static lifetime for Hnsw.
        // We store the raw pointer to reconstruct and drop it later.
        let io_box = Box::new(HnswIo::new(dir, "hnsw"));
        let io_ptr = Box::into_raw(io_box);
        // Safety: io_ptr is valid, we just allocated it
        let io_ref: &'static mut HnswIo = unsafe { &mut *io_ptr };

        let inner = match metric {
            DistanceMetric::L2 => {
                let hnsw: Hnsw<'static, f32, DistL2> = io_ref
                    .load_hnsw()
                    .map_err(|e| VdbError::Index(format!("failed to load HNSW: {e}")))?;
                HnswInner::L2(hnsw)
            }
            DistanceMetric::Cosine | DistanceMetric::InnerProduct => {
                let hnsw: Hnsw<'static, f32, DistDot> = io_ref
                    .load_hnsw()
                    .map_err(|e| VdbError::Index(format!("failed to load HNSW: {e}")))?;
                HnswInner::Dot(hnsw)
            }
        };

        // Safety: io_ptr was allocated from Box::into_raw and is still valid
        let io_box = unsafe { Box::from_raw(io_ptr) };

        Ok(Self {
            dimension,
            metric,
            ef_search: config.ef_search,
            inner: RwLock::new(inner),
            _io: Some(io_box),
        })
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&self, ids: &[u64], vectors: &[&[f32]]) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(VdbError::InvalidVector(
                "ids and vectors length mismatch".to_string(),
            ));
        }
        for v in vectors.iter() {
            if v.len() != self.dimension {
                return Err(VdbError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len(),
                });
            }
        }

        let owned_vecs: Vec<Vec<f32>> = if self.metric == DistanceMetric::Cosine {
            vectors.iter().map(|v| normalize(v)).collect()
        } else {
            vectors.iter().map(|v| v.to_vec()).collect()
        };

        let data_with_ids: Vec<(&Vec<f32>, usize)> = owned_vecs
            .iter()
            .zip(ids.iter())
            .map(|(v, id)| (v, *id as usize))
            .collect();

        match &*self.inner.write() {
            HnswInner::L2(hnsw) => hnsw.parallel_insert(&data_with_ids),
            HnswInner::Dot(hnsw) => hnsw.parallel_insert(&data_with_ids),
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(VdbError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let ef = if ef > 0 { ef } else { self.ef_search };

        let query_vec = if self.metric == DistanceMetric::Cosine {
            normalize(query)
        } else {
            query.to_vec()
        };

        let neighbours = match &*self.inner.read() {
            HnswInner::L2(hnsw) => hnsw.search(&query_vec, k, ef),
            HnswInner::Dot(hnsw) => hnsw.search(&query_vec, k, ef),
        };

        let mut results: Vec<SearchResult> = neighbours
            .into_iter()
            .map(|n| {
                let dist = if self.metric == DistanceMetric::Cosine {
                    // DistDot returns 1 - dot(a,b) for normalized vectors
                    // This already equals cosine distance
                    n.distance
                } else {
                    n.distance
                };
                (n.d_id as u64, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    fn len(&self) -> usize {
        match &*self.inner.read() {
            HnswInner::L2(hnsw) => hnsw.get_nb_point(),
            HnswInner::Dot(hnsw) => hnsw.get_nb_point(),
        }
    }

    fn save(&self) -> Result<Vec<u8>> {
        Err(VdbError::Internal(
            "use save_to_dir for HNSW index".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdb_common::config::IndexConfig;

    fn test_config() -> IndexConfig {
        IndexConfig {
            max_nb_connection: 16,
            max_elements: 1000,
            ef_construction: 200,
            ef_search: 50,
        }
    }

    #[test]
    fn test_hnsw_insert_search() {
        let idx = HnswIndex::new(3, DistanceMetric::L2, &test_config());

        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.9f32, 0.1, 0.0];
        idx.insert(&[0, 1, 2], &[&v1, &v2, &v3]).unwrap();

        let query = [1.0, 0.0, 0.0];
        let results = idx.search(&query, 2, 50).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hnsw_cosine() {
        let idx = HnswIndex::new(3, DistanceMetric::Cosine, &test_config());

        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [0.9f32, 0.1, 0.0];
        idx.insert(&[0, 1, 2], &[&v1, &v2, &v3]).unwrap();

        let query = [1.0, 0.0, 0.0];
        let results = idx.search(&query, 2, 50).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hnsw_save_load() {
        let idx = HnswIndex::new(3, DistanceMetric::L2, &test_config());

        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        idx.insert(&[0, 1], &[&v1, &v2]).unwrap();

        let dir = tempfile::tempdir().unwrap();
        idx.save_to_dir(dir.path()).unwrap();

        let loaded =
            HnswIndex::load_from_dir(3, DistanceMetric::L2, &test_config(), dir.path()).unwrap();
        assert_eq!(loaded.len(), 2);

        let query = [1.0, 0.0, 0.0];
        let results = loaded.search(&query, 1, 50).unwrap();
        assert_eq!(results[0].0, 0);
    }
}
