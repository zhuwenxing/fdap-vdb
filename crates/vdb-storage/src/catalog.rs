use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vdb_common::config::CollectionConfig;
use vdb_common::error::{Result, VdbError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: String,
    pub num_rows: usize,
    pub has_index: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub config: CollectionConfig,
    pub segments: Vec<SegmentMeta>,
    pub next_row_id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Catalog {
    pub collections: HashMap<String, CollectionMeta>,
    #[serde(skip)]
    path: PathBuf,
}

impl Catalog {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let path = data_dir.join("catalog.json");
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let mut catalog: Catalog = serde_json::from_str(&content)?;
            catalog.path = path;
            Ok(catalog)
        } else {
            std::fs::create_dir_all(data_dir)?;
            Ok(Catalog {
                collections: HashMap::new(),
                path,
            })
        }
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&self.path, content)?;
        Ok(())
    }

    pub fn create_collection(&mut self, config: CollectionConfig) -> Result<()> {
        if self.collections.contains_key(&config.name) {
            return Err(VdbError::CollectionAlreadyExists(config.name.clone()));
        }
        let meta = CollectionMeta {
            config,
            segments: Vec::new(),
            next_row_id: 0,
        };
        self.collections.insert(meta.config.name.clone(), meta);
        self.save()
    }

    pub fn drop_collection(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_none() {
            return Err(VdbError::CollectionNotFound(name.to_string()));
        }
        self.save()
    }

    pub fn get_collection(&self, name: &str) -> Result<&CollectionMeta> {
        self.collections
            .get(name)
            .ok_or_else(|| VdbError::CollectionNotFound(name.to_string()))
    }

    pub fn get_collection_mut(&mut self, name: &str) -> Result<&mut CollectionMeta> {
        self.collections
            .get_mut(name)
            .ok_or_else(|| VdbError::CollectionNotFound(name.to_string()))
    }

    pub fn register_segment(&mut self, collection: &str, segment: SegmentMeta) -> Result<()> {
        let meta = self.get_collection_mut(collection)?;
        meta.segments.push(segment);
        self.save()
    }

    pub fn advance_row_id(&mut self, collection: &str, count: u64) -> Result<u64> {
        let meta = self.get_collection_mut(collection)?;
        let start = meta.next_row_id;
        meta.next_row_id += count;
        self.save()?;
        Ok(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdb_common::metrics::DistanceMetric;

    #[test]
    fn test_catalog_crud() {
        let dir = tempfile::tempdir().unwrap();
        let mut catalog = Catalog::open(dir.path()).unwrap();

        let config = CollectionConfig {
            name: "test".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            index_config: Default::default(),
            metadata_fields: vec![],
            storage_format: Default::default(),
        };
        catalog.create_collection(config.clone()).unwrap();
        assert!(catalog.get_collection("test").is_ok());

        // duplicate
        assert!(catalog.create_collection(config).is_err());

        catalog.drop_collection("test").unwrap();
        assert!(catalog.get_collection("test").is_err());
    }
}
