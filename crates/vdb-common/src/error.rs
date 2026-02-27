use thiserror::Error;

#[derive(Error, Debug)]
pub enum VdbError {
    #[error("collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid schema: {0}")]
    InvalidSchema(String),

    #[error("invalid vector: {0}")]
    InvalidVector(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("query error: {0}")]
    Query(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("vortex error: {0}")]
    Vortex(String),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, VdbError>;
