use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::pyarrow::ToPyArrow;
use numpy::PyArrayLike1;
use numpy::PyArrayLike2;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use vdb_common::config::{CollectionConfig, MetadataFieldConfig, MetadataFieldType};
use vdb_common::error::VdbError;
use vdb_common::metrics::DistanceMetric;
use vdb_query::context::create_session_context;
use vdb_storage::engine::{MetadataColumnValue, StorageEngine};

/// Convert VdbError to PyErr.
fn vdb_err(e: VdbError) -> PyErr {
    match e {
        VdbError::CollectionNotFound(s) => PyKeyError::new_err(s),
        VdbError::CollectionAlreadyExists(s) => PyValueError::new_err(s),
        VdbError::DimensionMismatch { expected, actual } => PyValueError::new_err(format!(
            "dimension mismatch: expected {expected}, got {actual}"
        )),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Extract vectors from Python object: numpy 2D array or list[list[float]].
fn extract_vectors(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    // Try numpy 2D array first
    if let Ok(arr) = obj.extract::<PyArrayLike2<f32>>() {
        let view = arr.as_array();
        let rows = view.nrows();
        let cols = view.ncols();
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(view[[i, j]]);
            }
            result.push(row);
        }
        return Ok(result);
    }

    // Fallback: list[list[float]]
    let list = obj.downcast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());
    for item in list.iter() {
        let inner: Vec<f32> = item.extract()?;
        result.push(inner);
    }
    Ok(result)
}

/// Extract a single query vector from Python object: numpy 1D array or list[float].
fn extract_query_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try numpy 1D array
    if let Ok(arr) = obj.extract::<PyArrayLike1<f32>>() {
        return Ok(arr.as_array().to_vec());
    }

    // Fallback: list[float]
    obj.extract::<Vec<f32>>()
}

/// Convert Python metadata dict to engine MetadataColumnValue map.
fn extract_metadata(
    obj: Option<&Bound<'_, PyDict>>,
    n: usize,
) -> PyResult<HashMap<String, Vec<MetadataColumnValue>>> {
    let Some(dict) = obj else {
        return Ok(HashMap::new());
    };
    let mut result = HashMap::new();
    for (key, value) in dict.iter() {
        let field_name: String = key.extract()?;
        let list = value.downcast::<PyList>()?;
        if list.len() != n {
            return Err(PyValueError::new_err(format!(
                "metadata field '{}' has {} values, expected {}",
                field_name,
                list.len(),
                n
            )));
        }
        let mut col_values = Vec::with_capacity(n);
        for item in list.iter() {
            if item.is_none() {
                col_values.push(MetadataColumnValue::Null);
            } else if item.is_instance_of::<PyBool>() {
                // Must check bool before int (isinstance(True, int) == True in Python)
                col_values.push(MetadataColumnValue::Bool(item.extract::<bool>()?));
            } else if item.is_instance_of::<PyInt>() {
                col_values.push(MetadataColumnValue::Int64(item.extract::<i64>()?));
            } else if item.is_instance_of::<PyFloat>() {
                col_values.push(MetadataColumnValue::Float64(item.extract::<f64>()?));
            } else if item.is_instance_of::<PyString>() {
                col_values.push(MetadataColumnValue::String(item.extract::<String>()?));
            } else {
                return Err(PyTypeError::new_err(format!(
                    "unsupported metadata type for field '{}'",
                    field_name
                )));
            }
        }
        result.insert(field_name, col_values);
    }
    Ok(result)
}

/// Parse distance metric string.
fn parse_metric(s: Option<&str>) -> PyResult<DistanceMetric> {
    match s {
        None => Ok(DistanceMetric::default()),
        Some(s) => s
            .parse::<DistanceMetric>()
            .map_err(PyValueError::new_err),
    }
}

/// Parse metadata field type string.
fn parse_field_type(s: &str) -> PyResult<MetadataFieldType> {
    s.parse::<MetadataFieldType>()
        .map_err(PyValueError::new_err)
}

#[pyclass]
#[allow(clippy::upper_case_acronyms)]
struct VDB {
    engine: Arc<StorageEngine>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl VDB {
    /// Open or create a database at the given directory.
    #[new]
    fn new(data_dir: &str) -> PyResult<Self> {
        let path = Path::new(data_dir);
        std::fs::create_dir_all(path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create data dir: {e}")))?;
        let engine = StorageEngine::open(path).map_err(vdb_err)?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create tokio runtime: {e}")))?;
        Ok(Self {
            engine: Arc::new(engine),
            runtime,
        })
    }

    /// Create a new collection.
    #[pyo3(signature = (name, dimension, distance_metric=None, metadata_fields=None))]
    fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance_metric: Option<&str>,
        metadata_fields: Option<Vec<(String, String)>>,
    ) -> PyResult<()> {
        let metric = parse_metric(distance_metric)?;
        let fields = metadata_fields
            .unwrap_or_default()
            .into_iter()
            .map(|(n, t)| {
                Ok(MetadataFieldConfig {
                    name: n,
                    field_type: parse_field_type(&t)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let config = CollectionConfig {
            name: name.to_string(),
            dimension,
            distance_metric: metric,
            index_config: Default::default(),
            metadata_fields: fields,
        };
        self.engine.create_collection(config).map_err(vdb_err)
    }

    /// List all collections.
    fn list_collections<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let configs = self.engine.list_collections();
        let mut result = Vec::with_capacity(configs.len());
        for c in configs {
            let dict = PyDict::new(py);
            dict.set_item("name", &c.name)?;
            dict.set_item("dimension", c.dimension)?;
            dict.set_item("distance_metric", c.distance_metric.as_str())?;
            let fields: Vec<(&str, &str)> = c
                .metadata_fields
                .iter()
                .map(|f| {
                    let type_str = match f.field_type {
                        MetadataFieldType::String => "string",
                        MetadataFieldType::Int64 => "int64",
                        MetadataFieldType::Float64 => "float64",
                        MetadataFieldType::Bool => "bool",
                    };
                    (f.name.as_str(), type_str)
                })
                .collect();
            dict.set_item("metadata_fields", fields)?;
            result.push(dict);
        }
        Ok(result)
    }

    /// Drop a collection.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.engine.drop_collection(name).map_err(vdb_err)
    }

    /// Insert vectors with IDs and optional metadata.
    ///
    /// `vectors` accepts a numpy 2D array or list[list[float]].
    /// `metadata` is an optional dict mapping field name to list of values.
    #[pyo3(signature = (collection, ids, vectors, metadata=None))]
    fn insert(
        &self,
        collection: &str,
        ids: Vec<String>,
        vectors: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let vecs = extract_vectors(vectors)?;
        if ids.len() != vecs.len() {
            return Err(PyValueError::new_err(format!(
                "ids length ({}) != vectors length ({})",
                ids.len(),
                vecs.len()
            )));
        }
        let meta = extract_metadata(metadata, ids.len())?;
        self.engine
            .insert(collection, ids, vecs, meta)
            .map_err(vdb_err)
    }

    /// Search for similar vectors.
    ///
    /// `query_vector` accepts a numpy 1D array or list[float].
    /// Returns a list of dicts with keys: id, distance, metadata.
    #[pyo3(signature = (collection, query_vector, top_k=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        collection: &str,
        query_vector: &Bound<'_, PyAny>,
        top_k: Option<usize>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let query = extract_query_vector(query_vector)?;
        let k = top_k.unwrap_or(10);
        let ef = k.max(50); // ef >= top_k for good recall
        let hits = self.engine.search(collection, &query, k, ef).map_err(vdb_err)?;

        let mut results = Vec::with_capacity(hits.len());
        for hit in hits {
            let dict = PyDict::new(py);
            dict.set_item("id", &hit.id)?;
            dict.set_item("distance", hit.distance)?;
            if !hit.metadata.is_empty() {
                let meta_dict = PyDict::new(py);
                for (k, v) in &hit.metadata {
                    meta_dict.set_item(k, v)?;
                }
                dict.set_item("metadata", meta_dict)?;
            } else {
                dict.set_item("metadata", PyDict::new(py))?;
            }
            results.push(dict);
        }
        Ok(results)
    }

    /// Delete vectors by IDs.
    fn delete(&self, collection: &str, ids: Vec<String>) -> PyResult<u64> {
        self.engine.delete(collection, &ids).map_err(vdb_err)
    }

    /// Flush memtable to disk.
    fn flush(&self, collection: &str) -> PyResult<()> {
        self.engine.flush(collection).map_err(vdb_err)
    }

    /// Compact a collection: merge segments and remove deleted rows.
    ///
    /// Returns a dict with keys: segments_before, segments_after, rows_removed.
    fn compact<'py>(&self, py: Python<'py>, collection: &str) -> PyResult<Bound<'py, PyDict>> {
        let (before, after, removed) = self.engine.compact(collection).map_err(vdb_err)?;
        let dict = PyDict::new(py);
        dict.set_item("segments_before", before)?;
        dict.set_item("segments_after", after)?;
        dict.set_item("rows_removed", removed)?;
        Ok(dict)
    }

    /// Execute a SQL query against the database.
    ///
    /// Returns a pyarrow.RecordBatch.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyAny>> {
        let engine = self.engine.clone();
        let ctx = create_session_context(engine)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let batches = self.runtime.block_on(async {
            let df = ctx
                .sql(query)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            df.collect()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;

        if batches.is_empty() {
            return Err(PyRuntimeError::new_err("query returned no batches"));
        }

        let schema = batches[0].schema();
        let merged = concat_batches(&schema, &batches)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        merged
            .to_pyarrow(py)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymodule]
fn fdap_vdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VDB>()?;
    Ok(())
}
