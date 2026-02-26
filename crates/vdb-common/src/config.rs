use serde::{Deserialize, Serialize};

use crate::metrics::DistanceMetric;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    #[serde(default)]
    pub distance_metric: DistanceMetric,
    #[serde(default)]
    pub index_config: IndexConfig,
    #[serde(default)]
    pub metadata_fields: Vec<MetadataFieldConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFieldConfig {
    pub name: String,
    pub field_type: MetadataFieldType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataFieldType {
    String,
    Int64,
    Float64,
    Bool,
}

impl std::str::FromStr for MetadataFieldType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "string" | "utf8" => Ok(MetadataFieldType::String),
            "int64" | "int" | "integer" => Ok(MetadataFieldType::Int64),
            "float64" | "float" | "double" => Ok(MetadataFieldType::Float64),
            "bool" | "boolean" => Ok(MetadataFieldType::Bool),
            _ => Err(format!("unknown field type: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub max_nb_connection: usize,
    pub max_elements: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_elements: 1_000_000,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemtableConfig {
    pub max_rows: usize,
    pub max_bytes: usize,
    pub flush_interval_secs: u64,
}

impl Default for MemtableConfig {
    fn default() -> Self {
        Self {
            max_rows: 100_000,
            max_bytes: 256 * 1024 * 1024,
            flush_interval_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Check interval in seconds.
    pub check_interval_secs: u64,
    /// Trigger compact when segment count >= this threshold.
    pub max_segments: usize,
    /// Trigger compact when deleted ratio > this threshold (0.0 ~ 1.0).
    pub delete_ratio_threshold: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 60,
            max_segments: 4,
            delete_ratio_threshold: 0.2,
        }
    }
}
