use parking_lot::RwLock;
use vdb_common::error::{Result, VdbError};
use vdb_common::metrics::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

/// Brute-force vector index for testing and small datasets.
pub struct FlatIndex {
    dimension: usize,
    metric: DistanceMetric,
    data: RwLock<FlatData>,
}

struct FlatData {
    ids: Vec<u64>,
    vectors: Vec<f32>, // flattened: len = ids.len() * dimension
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            dimension,
            metric,
            data: RwLock::new(FlatData {
                ids: Vec::new(),
                vectors: Vec::new(),
            }),
        }
    }

    pub fn from_data(
        dimension: usize,
        metric: DistanceMetric,
        ids: Vec<u64>,
        vectors: Vec<f32>,
    ) -> Result<Self> {
        if vectors.len() != ids.len() * dimension {
            return Err(VdbError::DimensionMismatch {
                expected: ids.len() * dimension,
                actual: vectors.len(),
            });
        }
        Ok(Self {
            dimension,
            metric,
            data: RwLock::new(FlatData { ids, vectors }),
        })
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&self, ids: &[u64], vectors: &[&[f32]]) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(VdbError::InvalidVector(
                "ids and vectors length mismatch".to_string(),
            ));
        }
        let mut data = self.data.write();
        for (id, vec) in ids.iter().zip(vectors.iter()) {
            if vec.len() != self.dimension {
                return Err(VdbError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vec.len(),
                });
            }
            data.ids.push(*id);
            data.vectors.extend_from_slice(vec);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, _ef: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(VdbError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let data = self.data.read();
        let n = data.ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut results: Vec<SearchResult> = (0..n)
            .map(|i| {
                let start = i * self.dimension;
                let end = start + self.dimension;
                let vec = &data.vectors[start..end];
                let dist = self.metric.compute(query, vec);
                (data.ids[i], dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    fn len(&self) -> usize {
        self.data.read().ids.len()
    }

    fn save(&self) -> Result<Vec<u8>> {
        let data = self.data.read();
        let serialized = serde_json::to_vec(&(&data.ids, &data.vectors, self.dimension))
            .map_err(|e| VdbError::Internal(e.to_string()))?;
        Ok(serialized)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_index_search() {
        let idx = FlatIndex::new(3, DistanceMetric::L2);
        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let v3 = [1.0f32, 1.0, 0.0];
        idx.insert(&[0, 1, 2], &[&v1, &v2, &v3]).unwrap();

        let query = [1.0, 0.0, 0.0];
        let results = idx.search(&query, 2, 0).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // exact match first
        assert!(results[0].1 < results[1].1);
    }
}
