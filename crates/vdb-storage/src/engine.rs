use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, StringArray,
    TimestampMicrosecondArray,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use parking_lot::RwLock;
use vdb_common::config::CollectionConfig;
use vdb_common::error::{Result, VdbError};
use vdb_common::schema::{self, COL_CREATED_AT, COL_DELETED, COL_ID, COL_VECTOR};
use vdb_index::traits::VectorIndex;

use crate::catalog::{Catalog, SegmentMeta};
use crate::memtable::MemTable;
use crate::segment::Segment;
use crate::wal::Wal;

struct CollectionState {
    config: CollectionConfig,
    schema: SchemaRef,
    memtable: MemTable,
    wal: Wal,
    segments: Vec<Segment>,
    deleted_ids: HashSet<String>,
}

pub struct StorageEngine {
    data_dir: PathBuf,
    catalog: RwLock<Catalog>,
    collections: RwLock<HashMap<String, Arc<RwLock<CollectionState>>>>,
}

impl StorageEngine {
    /// Path to the deleted IDs file for a collection.
    fn deleted_ids_path(data_dir: &Path, collection: &str) -> PathBuf {
        data_dir
            .join("collections")
            .join(collection)
            .join("deleted_ids.json")
    }

    /// Load deleted IDs from disk.
    fn load_deleted_ids(data_dir: &Path, collection: &str) -> HashSet<String> {
        let path = Self::deleted_ids_path(data_dir, collection);
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(ids) = serde_json::from_str::<Vec<String>>(&content) {
                    return ids.into_iter().collect();
                }
            }
        }
        HashSet::new()
    }

    /// Persist deleted IDs to disk.
    fn save_deleted_ids(data_dir: &Path, collection: &str, ids: &HashSet<String>) -> Result<()> {
        let path = Self::deleted_ids_path(data_dir, collection);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let vec: Vec<&String> = ids.iter().collect();
        let content = serde_json::to_string(&vec)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn open(data_dir: &Path) -> Result<Self> {
        let catalog = Catalog::open(data_dir)?;
        let mut collections = HashMap::new();

        for (name, meta) in &catalog.collections {
            let schema = Arc::new(schema::collection_schema(&meta.config)?);
            let wal = Wal::new(data_dir, name)?;

            let seg_base = data_dir.join("collections").join(name).join("segments");
            let mut segments = Vec::new();
            for seg_meta in &meta.segments {
                match Segment::load(&seg_base, &seg_meta.id, &meta.config) {
                    Ok(seg) => segments.push(seg),
                    Err(e) => tracing::warn!("failed to load segment {}: {e}", seg_meta.id),
                }
            }

            let memtable = MemTable::new(schema.clone());
            let wal_batches = wal.read_all(&schema)?;
            for batch in wal_batches {
                memtable.insert(batch);
            }
            if !memtable.is_empty() {
                tracing::info!(
                    "recovered {} rows from WAL for collection {name}",
                    memtable.row_count()
                );
            }

            let deleted_ids = Self::load_deleted_ids(data_dir, name);
            if !deleted_ids.is_empty() {
                tracing::info!(
                    "loaded {} deleted IDs for collection {name}",
                    deleted_ids.len()
                );
            }

            let state = CollectionState {
                config: meta.config.clone(),
                schema,
                memtable,
                wal,
                segments,
                deleted_ids,
            };
            collections.insert(name.clone(), Arc::new(RwLock::new(state)));
        }

        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            catalog: RwLock::new(catalog),
            collections: RwLock::new(collections),
        })
    }

    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        let schema = Arc::new(schema::collection_schema(&config)?);
        self.catalog.write().create_collection(config.clone())?;

        let wal = Wal::new(&self.data_dir, &config.name)?;
        let memtable = MemTable::new(schema.clone());

        let state = CollectionState {
            config,
            schema,
            memtable,
            wal,
            segments: Vec::new(),
            deleted_ids: HashSet::new(),
        };
        let name = state.config.name.clone();
        self.collections
            .write()
            .insert(name, Arc::new(RwLock::new(state)));
        Ok(())
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.catalog.write().drop_collection(name)?;
        self.collections.write().remove(name);

        let col_dir = self.data_dir.join("collections").join(name);
        if col_dir.exists() {
            let _ = std::fs::remove_dir_all(&col_dir);
        }
        let wal_dir = self.data_dir.join("wal").join(name);
        if wal_dir.exists() {
            let _ = std::fs::remove_dir_all(&wal_dir);
        }
        Ok(())
    }

    pub fn list_collections(&self) -> Vec<CollectionConfig> {
        let catalog = self.catalog.read();
        catalog
            .collections
            .values()
            .map(|m| m.config.clone())
            .collect()
    }

    pub fn get_collection_config(&self, name: &str) -> Result<CollectionConfig> {
        let catalog = self.catalog.read();
        Ok(catalog.get_collection(name)?.config.clone())
    }

    pub fn get_collection_schema(&self, name: &str) -> Result<SchemaRef> {
        let state_arc = self.get_state(name)?;
        let guard = state_arc.read();
        let schema = guard.schema.clone();
        Ok(schema)
    }

    fn get_state(&self, collection: &str) -> Result<Arc<RwLock<CollectionState>>> {
        let cols = self.collections.read();
        cols.get(collection)
            .cloned()
            .ok_or_else(|| VdbError::CollectionNotFound(collection.to_string()))
    }

    /// Insert vectors with IDs and optional metadata.
    pub fn insert(
        &self,
        collection: &str,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: HashMap<String, Vec<MetadataColumnValue>>,
    ) -> Result<u64> {
        let state_arc = self.get_state(collection)?;
        let state = state_arc.read();
        let n = ids.len();
        let dim = state.config.dimension;

        for v in &vectors {
            if v.len() != dim {
                return Err(VdbError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }

        let batch = build_record_batch(&state.schema, &state.config, ids, vectors, metadata)?;
        state.wal.append(&batch)?;
        state.memtable.insert(batch);

        Ok(n as u64)
    }

    /// Flush memtable to a new segment.
    pub fn flush(&self, collection: &str) -> Result<()> {
        let state_arc = self.get_state(collection)?;
        let mut state = state_arc.write();
        let batches = state.memtable.freeze();
        if batches.is_empty() {
            return Ok(());
        }

        let seg_id = uuid::Uuid::new_v4().to_string();
        let seg_base = self
            .data_dir
            .join("collections")
            .join(collection)
            .join("segments");

        let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let segment = Segment::write(
            &seg_base,
            &seg_id,
            state.schema.clone(),
            &batches,
            &state.config,
        )?;
        state.segments.push(segment);

        self.catalog.write().register_segment(
            collection,
            SegmentMeta {
                id: seg_id,
                num_rows,
                has_index: true,
            },
        )?;

        state.wal.clear()?;
        tracing::info!("flushed {num_rows} rows to new segment for collection {collection}");
        Ok(())
    }

    /// Search for similar vectors across all segments and memtable.
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> Result<Vec<SearchHit>> {
        let state_arc = self.get_state(collection)?;
        let state = state_arc.read();

        let dim = state.config.dimension;
        if query.len() != dim {
            return Err(VdbError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }

        let mut all_hits = Vec::new();

        // Search segments (HNSW index → selective Parquet read)
        for segment in &state.segments {
            let mut results = segment.index.search(query, top_k, ef)?;
            if results.is_empty() {
                continue;
            }

            // Sort by row_id so we can use Parquet RowSelection
            results.sort_unstable_by_key(|(id, _)| *id);

            let sorted_row_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let sorted_distances: Vec<f32> = results.iter().map(|(_, d)| *d).collect();

            // Read only the rows we need, excluding the vector column
            let batch = segment.read_metadata_by_row_ids(&sorted_row_ids)?;

            let id_col = batch
                .column_by_name(COL_ID)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let deleted_col = batch
                .column_by_name(COL_DELETED)
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

            for (i, dist) in sorted_distances.iter().enumerate() {
                if i >= batch.num_rows() {
                    break;
                }
                let is_deleted = deleted_col.is_some_and(|d| d.value(i));
                if is_deleted {
                    continue;
                }
                let id = id_col.map_or_else(String::new, |c| c.value(i).to_string());
                if state.deleted_ids.contains(&id) {
                    continue;
                }

                let meta = extract_metadata(&batch, i);
                all_hits.push(SearchHit {
                    id,
                    distance: *dist,
                    metadata: meta,
                });
            }
        }

        // Search memtable (brute force)
        let mem_batches = state.memtable.batches();
        for batch in &mem_batches {
            let metric = state.config.distance_metric;
            let vector_col = batch.column_by_name(COL_VECTOR);
            let id_col = batch
                .column_by_name(COL_ID)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let deleted_col = batch
                .column_by_name(COL_DELETED)
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

            if let (Some(vc), Some(ic)) = (vector_col, id_col) {
                if let Some(fsl) = vc.as_any().downcast_ref::<FixedSizeListArray>() {
                    let values = fsl
                        .values()
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap();
                    let raw = values.values();

                    for i in 0..batch.num_rows() {
                        let is_deleted = deleted_col.is_some_and(|d| d.value(i));
                        if is_deleted {
                            continue;
                        }
                        let row_id = ic.value(i).to_string();
                        if state.deleted_ids.contains(&row_id) {
                            continue;
                        }
                        let start = i * dim;
                        let end = start + dim;
                        let vec = &raw[start..end];
                        let dist = metric.compute(query, vec);

                        let meta = extract_metadata(batch, i);
                        all_hits.push(SearchHit {
                            id: row_id,
                            distance: dist,
                            metadata: meta,
                        });
                    }
                }
            }
        }

        all_hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hits.truncate(top_k);

        Ok(all_hits)
    }

    /// Get all segments and memtable data for a collection (for DataFusion TableProvider).
    pub fn get_collection_data(&self, collection: &str) -> Result<Vec<RecordBatch>> {
        let state_arc = self.get_state(collection)?;
        let state = state_arc.read();
        let mut all_batches = Vec::new();

        for segment in &state.segments {
            if let Ok(batch) = segment.read_all() {
                all_batches.push(batch);
            }
        }

        all_batches.extend(state.memtable.batches());
        Ok(all_batches)
    }

    /// Check if memtable should be flushed.
    pub fn maybe_flush(&self, collection: &str, max_rows: usize) -> Result<bool> {
        let state_arc = self.get_state(collection)?;
        let should = state_arc.read().memtable.should_flush(max_rows);
        if should {
            self.flush(collection)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Soft-delete vectors by their IDs.
    ///
    /// Returns the number of IDs that were newly marked as deleted.
    pub fn delete(&self, collection: &str, ids: &[String]) -> Result<u64> {
        let state_arc = self.get_state(collection)?;
        let mut state = state_arc.write();

        let mut count = 0u64;
        for id in ids {
            if state.deleted_ids.insert(id.clone()) {
                count += 1;
            }
        }

        if count > 0 {
            Self::save_deleted_ids(&self.data_dir, collection, &state.deleted_ids)?;
            tracing::info!("soft-deleted {count} IDs from collection {collection}");
        }

        Ok(count)
    }

    /// Compact a collection: merge all segments, remove deleted rows, produce a single new segment.
    ///
    /// Returns `(segments_before, segments_after, rows_removed)`.
    pub fn compact(&self, collection: &str) -> Result<(usize, usize, usize)> {
        let state_arc = self.get_state(collection)?;
        let mut state = state_arc.write();

        // First flush memtable so all data is in segments
        let mem_batches = state.memtable.freeze();
        if !mem_batches.is_empty() {
            let seg_id = uuid::Uuid::new_v4().to_string();
            let seg_base = self
                .data_dir
                .join("collections")
                .join(collection)
                .join("segments");
            let num_rows: usize = mem_batches.iter().map(|b| b.num_rows()).sum();
            let segment = Segment::write(
                &seg_base,
                &seg_id,
                state.schema.clone(),
                &mem_batches,
                &state.config,
            )?;
            state.segments.push(segment);
            self.catalog.write().register_segment(
                collection,
                SegmentMeta {
                    id: seg_id,
                    num_rows,
                    has_index: true,
                },
            )?;
            state.wal.clear()?;
        }

        let segments_before = state.segments.len();
        if segments_before == 0 {
            return Ok((0, 0, 0));
        }

        // Read all segment data
        let mut all_batches = Vec::new();
        for segment in &state.segments {
            if let Ok(batch) = segment.read_all() {
                all_batches.push(batch);
            }
        }

        if all_batches.is_empty() {
            return Ok((segments_before, 0, 0));
        }

        let merged = concat_batches(&state.schema, &all_batches)?;
        let total_rows = merged.num_rows();

        // Filter out deleted rows
        let id_col = merged
            .column_by_name(COL_ID)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let deleted_col = merged
            .column_by_name(COL_DELETED)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

        let mut keep_indices = Vec::new();
        for i in 0..total_rows {
            let is_soft_deleted = deleted_col.is_some_and(|d| d.value(i));
            let is_id_deleted = id_col
                .is_some_and(|c| state.deleted_ids.contains(c.value(i)));
            if !is_soft_deleted && !is_id_deleted {
                keep_indices.push(i as u64);
            }
        }

        let rows_removed = total_rows - keep_indices.len();

        // Remove old segment directories
        let seg_base = self
            .data_dir
            .join("collections")
            .join(collection)
            .join("segments");
        let old_seg_ids: Vec<String> = state.segments.iter().map(|s| s.id.clone()).collect();
        for seg_id in &old_seg_ids {
            let seg_dir = seg_base.join(seg_id);
            if seg_dir.exists() {
                let _ = std::fs::remove_dir_all(&seg_dir);
            }
        }
        state.segments.clear();

        // Write new compacted segment if there are remaining rows
        let segments_after;
        if !keep_indices.is_empty() {
            let indices_array = arrow::array::UInt64Array::from(keep_indices);
            let columns: Vec<_> = merged
                .columns()
                .iter()
                .map(|col| arrow::compute::take(col, &indices_array, None))
                .collect::<std::result::Result<_, _>>()?;
            let filtered = RecordBatch::try_new(state.schema.clone(), columns)?;

            let seg_id = uuid::Uuid::new_v4().to_string();
            let num_rows = filtered.num_rows();
            let segment = Segment::write(
                &seg_base,
                &seg_id,
                state.schema.clone(),
                &[filtered],
                &state.config,
            )?;
            state.segments.push(segment);

            // Update catalog: replace all segments with one
            {
                let mut catalog = self.catalog.write();
                let meta = catalog.get_collection_mut(collection)?;
                meta.segments = vec![SegmentMeta {
                    id: seg_id,
                    num_rows,
                    has_index: true,
                }];
                catalog.save()?;
            }
            segments_after = 1;
        } else {
            // All rows deleted, no segments remain
            let mut catalog = self.catalog.write();
            let meta = catalog.get_collection_mut(collection)?;
            meta.segments.clear();
            catalog.save()?;
            segments_after = 0;
        }

        // Clear deleted IDs since they're now physically removed
        state.deleted_ids.clear();
        Self::save_deleted_ids(&self.data_dir, collection, &state.deleted_ids)?;

        tracing::info!(
            "compacted collection {collection}: {segments_before} → {segments_after} segments, {rows_removed} rows removed"
        );

        Ok((segments_before, segments_after, rows_removed))
    }
}

fn extract_metadata(batch: &RecordBatch, i: usize) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    for field in batch.schema().fields() {
        let name = field.name().as_str();
        if name == COL_ID || name == COL_VECTOR || name == COL_CREATED_AT || name == COL_DELETED {
            continue;
        }
        if let Some(col) = batch.column_by_name(name) {
            if col.is_null(i) {
                continue;
            }
            if let Some(s) = col.as_any().downcast_ref::<StringArray>() {
                meta.insert(name.to_string(), s.value(i).to_string());
            } else if let Some(n) = col.as_any().downcast_ref::<Int64Array>() {
                meta.insert(name.to_string(), n.value(i).to_string());
            } else if let Some(f) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
                meta.insert(name.to_string(), f.value(i).to_string());
            } else if let Some(b) = col.as_any().downcast_ref::<BooleanArray>() {
                meta.insert(name.to_string(), b.value(i).to_string());
            }
        }
    }
    meta
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum MetadataColumnValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Null,
}

fn build_record_batch(
    schema: &SchemaRef,
    config: &CollectionConfig,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    metadata: HashMap<String, Vec<MetadataColumnValue>>,
) -> Result<RecordBatch> {
    let n = ids.len();

    let id_array: Arc<dyn arrow::array::Array> = Arc::new(StringArray::from(ids));

    let flat: Vec<f32> = vectors.into_iter().flatten().collect();
    let values = Float32Array::from(flat);
    let vector_array: Arc<dyn arrow::array::Array> = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            config.dimension as i32,
            Arc::new(values),
            None,
        )
        .map_err(VdbError::Arrow)?,
    );

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let timestamps: Vec<i64> = vec![now; n];
    let ts_array: Arc<dyn arrow::array::Array> =
        Arc::new(TimestampMicrosecondArray::from(timestamps));
    let deleted_array: Arc<dyn arrow::array::Array> = Arc::new(BooleanArray::from(vec![false; n]));

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![id_array, vector_array];

    // Build metadata columns in schema order
    for field in schema.fields() {
        let name = field.name().as_str();
        if name == COL_ID || name == COL_VECTOR || name == COL_CREATED_AT || name == COL_DELETED {
            continue;
        }
        let col: Arc<dyn arrow::array::Array> = if let Some(vals) = metadata.get(name) {
            match field.data_type() {
                DataType::Utf8 => {
                    let arr: Vec<Option<String>> = vals
                        .iter()
                        .map(|v| match v {
                            MetadataColumnValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    Arc::new(StringArray::from(arr))
                }
                DataType::Int64 => {
                    let arr: Vec<Option<i64>> = vals
                        .iter()
                        .map(|v| match v {
                            MetadataColumnValue::Int64(i) => Some(*i),
                            _ => None,
                        })
                        .collect();
                    Arc::new(Int64Array::from(arr))
                }
                DataType::Float64 => {
                    let arr: Vec<Option<f64>> = vals
                        .iter()
                        .map(|v| match v {
                            MetadataColumnValue::Float64(f) => Some(*f),
                            _ => None,
                        })
                        .collect();
                    Arc::new(arrow::array::Float64Array::from(arr))
                }
                DataType::Boolean => {
                    let arr: Vec<Option<bool>> = vals
                        .iter()
                        .map(|v| match v {
                            MetadataColumnValue::Bool(b) => Some(*b),
                            _ => None,
                        })
                        .collect();
                    Arc::new(BooleanArray::from(arr))
                }
                _ => Arc::new(StringArray::from(vec![None::<String>; n])),
            }
        } else {
            // No metadata provided for this column, fill with nulls
            match field.data_type() {
                DataType::Utf8 => Arc::new(StringArray::from(vec![None::<String>; n])),
                DataType::Int64 => Arc::new(Int64Array::from(vec![None::<i64>; n])),
                DataType::Float64 => {
                    Arc::new(arrow::array::Float64Array::from(vec![None::<f64>; n]))
                }
                DataType::Boolean => Arc::new(BooleanArray::from(vec![None::<bool>; n])),
                _ => Arc::new(StringArray::from(vec![None::<String>; n])),
            }
        };
        columns.push(col);
    }

    columns.push(ts_array);
    columns.push(deleted_array);

    RecordBatch::try_new(schema.clone(), columns).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vdb_common::config::{MetadataFieldConfig, MetadataFieldType};
    use vdb_common::metrics::DistanceMetric;

    fn test_config() -> CollectionConfig {
        CollectionConfig {
            name: "test".to_string(),
            dimension: 3,
            distance_metric: DistanceMetric::L2,
            index_config: Default::default(),
            metadata_fields: vec![MetadataFieldConfig {
                name: "category".to_string(),
                field_type: MetadataFieldType::String,
            }],
        }
    }

    #[test]
    fn test_engine_insert_search() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            vec![
                MetadataColumnValue::String("a".to_string()),
                MetadataColumnValue::String("b".to_string()),
            ],
        );

        engine
            .insert(
                "test",
                vec!["id1".to_string(), "id2".to_string()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                meta,
            )
            .unwrap();

        let hits = engine.search("test", &[1.0, 0.0, 0.0], 2, 50).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "id1");
    }

    #[test]
    fn test_engine_flush_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        engine
            .insert(
                "test",
                vec!["id1".to_string()],
                vec![vec![1.0, 0.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();

        engine.flush("test").unwrap();

        engine
            .insert(
                "test",
                vec!["id2".to_string()],
                vec![vec![0.0, 1.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();

        let hits = engine.search("test", &[1.0, 0.0, 0.0], 2, 50).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "id1");
    }

    #[test]
    fn test_delete_from_memtable() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        engine
            .insert(
                "test",
                vec!["id1".to_string(), "id2".to_string(), "id3".to_string()],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                HashMap::new(),
            )
            .unwrap();

        // Delete id2
        let count = engine.delete("test", &["id2".to_string()]).unwrap();
        assert_eq!(count, 1);

        // Search should not return id2
        let hits = engine.search("test", &[0.0, 1.0, 0.0], 10, 50).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(!hits.iter().any(|h| h.id == "id2"));

        // Deleting same ID again returns 0
        let count = engine.delete("test", &["id2".to_string()]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_delete_from_segment() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        engine
            .insert(
                "test",
                vec!["id1".to_string(), "id2".to_string()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();

        engine.flush("test").unwrap();

        // Delete from segment
        let count = engine.delete("test", &["id1".to_string()]).unwrap();
        assert_eq!(count, 1);

        let hits = engine.search("test", &[1.0, 0.0, 0.0], 10, 50).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "id2");
    }

    #[test]
    fn test_delete_persistence() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            engine.create_collection(test_config()).unwrap();
            engine
                .insert(
                    "test",
                    vec!["id1".to_string(), "id2".to_string()],
                    vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                    HashMap::new(),
                )
                .unwrap();
            engine.flush("test").unwrap();
            engine.delete("test", &["id1".to_string()]).unwrap();
        }

        // Reopen and verify deletion persisted
        let engine = StorageEngine::open(dir.path()).unwrap();
        let hits = engine.search("test", &[1.0, 0.0, 0.0], 10, 50).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "id2");
    }

    #[test]
    fn test_compact_removes_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        // Insert in two batches to create multiple segments
        engine
            .insert(
                "test",
                vec!["id1".to_string(), "id2".to_string()],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();
        engine.flush("test").unwrap();

        engine
            .insert(
                "test",
                vec!["id3".to_string(), "id4".to_string()],
                vec![vec![0.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();
        engine.flush("test").unwrap();

        // Delete two vectors
        engine
            .delete("test", &["id1".to_string(), "id3".to_string()])
            .unwrap();

        // Compact
        let (before, after, removed) = engine.compact("test").unwrap();
        assert_eq!(before, 2);
        assert_eq!(after, 1);
        assert_eq!(removed, 2);

        // Search should only find id2 and id4
        let hits = engine.search("test", &[1.0, 0.0, 0.0], 10, 50).unwrap();
        assert_eq!(hits.len(), 2);
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"id2"));
        assert!(ids.contains(&"id4"));
    }

    #[test]
    fn test_compact_merges_segments() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.create_collection(test_config()).unwrap();

        // Create 3 segments
        for i in 0..3 {
            engine
                .insert(
                    "test",
                    vec![format!("id{i}")],
                    vec![vec![i as f32, 0.0, 0.0]],
                    HashMap::new(),
                )
                .unwrap();
            engine.flush("test").unwrap();
        }

        let (before, after, removed) = engine.compact("test").unwrap();
        assert_eq!(before, 3);
        assert_eq!(after, 1);
        assert_eq!(removed, 0);

        // All vectors should still be searchable
        let hits = engine.search("test", &[1.0, 0.0, 0.0], 10, 50).unwrap();
        assert_eq!(hits.len(), 3);
    }
}
