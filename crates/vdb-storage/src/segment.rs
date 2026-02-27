use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{as_string_array, AsArray, BooleanArray, Float32Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow_schema::SchemaRef;
use vdb_common::config::{CollectionConfig, StorageFormat};
use vdb_common::error::{Result, VdbError};
use vdb_common::schema::{COL_DELETED, COL_ID, COL_VECTOR};
use vdb_index::hnsw::HnswIndex;
use vdb_index::traits::VectorIndex;

use crate::format;

pub struct Segment {
    pub id: String,
    path: PathBuf,
    pub index: Arc<HnswIndex>,
    pub schema: SchemaRef,
    pub num_rows: usize,
    format: StorageFormat,
}

impl Segment {
    /// Data file name for the given format.
    fn data_filename(format: StorageFormat) -> &'static str {
        match format {
            StorageFormat::Parquet => "data.parquet",
            StorageFormat::Vortex => "data.vortex",
        }
    }

    /// Write batches to a new segment on disk.
    pub fn write(
        base: &Path,
        seg_id: &str,
        schema: SchemaRef,
        batches: &[RecordBatch],
        config: &CollectionConfig,
    ) -> Result<Self> {
        if batches.is_empty() {
            return Err(VdbError::Storage("no batches to write".to_string()));
        }

        let seg_dir = base.join(seg_id);
        fs::create_dir_all(&seg_dir)?;

        // Merge all batches
        let merged = concat_batches(&schema, batches)?;
        let num_rows = merged.num_rows();

        let fmt = config.storage_format;
        let data_path = seg_dir.join(Self::data_filename(fmt));

        match fmt {
            StorageFormat::Parquet => {
                format::parquet::write_data(&data_path, schema.clone(), &merged)?;
            }
            StorageFormat::Vortex => {
                format::vortex::write_data(&data_path, schema.clone(), &merged)?;
            }
        }

        // Build HNSW index
        let index = build_hnsw_index(&merged, config)?;

        // Save index to directory
        let index_dir = seg_dir.join("hnsw_index");
        index.save_to_dir(&index_dir)?;

        Ok(Self {
            id: seg_id.to_string(),
            path: seg_dir,
            index: Arc::new(index),
            schema,
            num_rows,
            format: fmt,
        })
    }

    /// Load an existing segment from disk.
    pub fn load(base: &Path, seg_id: &str, config: &CollectionConfig) -> Result<Self> {
        let seg_dir = base.join(seg_id);
        let index_dir = seg_dir.join("hnsw_index");

        // Detect format by checking which data file exists
        let fmt = if seg_dir.join("data.vortex").exists() {
            StorageFormat::Vortex
        } else {
            StorageFormat::Parquet
        };

        let data_path = seg_dir.join(Self::data_filename(fmt));

        let (schema, num_rows) = match fmt {
            StorageFormat::Parquet => format::parquet::load_schema_and_rows(&data_path)?,
            StorageFormat::Vortex => {
                // For Vortex, read the first batch to get schema, use row_count for count
                let row_count = format::vortex::row_count(&data_path)? as usize;
                // Read a scan to get the arrow schema
                let schema = if row_count > 0 {
                    let batch = format::vortex::read_all(
                        &data_path,
                        &Arc::new(vdb_common::schema::collection_schema(config)?),
                    )?;
                    batch.schema()
                } else {
                    Arc::new(vdb_common::schema::collection_schema(config)?)
                };
                (schema, row_count)
            }
        };

        // Load HNSW index
        let index = HnswIndex::load_from_dir(
            config.dimension,
            config.distance_metric,
            &config.index_config,
            &index_dir,
        )?;

        Ok(Self {
            id: seg_id.to_string(),
            path: seg_dir,
            index: Arc::new(index),
            schema,
            num_rows,
            format: fmt,
        })
    }

    /// Read all data from the segment.
    pub fn read_all(&self) -> Result<RecordBatch> {
        let data_path = self.path.join(Self::data_filename(self.format));
        match self.format {
            StorageFormat::Parquet => format::parquet::read_all(&data_path, &self.schema),
            StorageFormat::Vortex => format::vortex::read_all(&data_path, &self.schema),
        }
    }

    /// Read rows by their IDs.
    pub fn read_by_ids(&self, ids: &[String]) -> Result<RecordBatch> {
        let all = self.read_all()?;
        let id_col = as_string_array(
            all.column_by_name(COL_ID)
                .ok_or_else(|| VdbError::InvalidSchema("missing _id column".to_string()))?,
        );
        let deleted_col = all
            .column_by_name(COL_DELETED)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

        let mut indices = Vec::new();
        for (i, id_val) in id_col.iter().enumerate() {
            if let Some(id_str) = id_val {
                let is_deleted = deleted_col.is_some_and(|d| d.value(i));
                if !is_deleted && ids.contains(&id_str.to_string()) {
                    indices.push(i as u64);
                }
            }
        }

        let indices_array = arrow::array::UInt64Array::from(indices);
        let columns: Vec<_> = all
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col, &indices_array, None))
            .collect::<std::result::Result<_, _>>()?;

        RecordBatch::try_new(all.schema(), columns).map_err(Into::into)
    }

    /// Read rows by internal row offsets (as returned by HNSW search).
    pub fn read_by_row_ids(&self, row_ids: &[u64]) -> Result<RecordBatch> {
        let all = self.read_all()?;
        let indices: Vec<u64> = row_ids
            .iter()
            .filter(|&&id| (id as usize) < all.num_rows())
            .copied()
            .collect();

        let indices_array = arrow::array::UInt64Array::from(indices);
        let columns: Vec<_> = all
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col, &indices_array, None))
            .collect::<std::result::Result<_, _>>()?;

        RecordBatch::try_new(all.schema(), columns).map_err(Into::into)
    }

    /// Efficiently read only non-vector columns for specific rows.
    ///
    /// `sorted_row_ids` must be sorted in ascending order.
    pub fn read_metadata_by_row_ids(&self, sorted_row_ids: &[u64]) -> Result<RecordBatch> {
        let data_path = self.path.join(Self::data_filename(self.format));
        match self.format {
            StorageFormat::Parquet => {
                format::parquet::read_metadata_by_row_ids(&data_path, &self.schema, sorted_row_ids)
            }
            StorageFormat::Vortex => {
                format::vortex::read_metadata_by_row_ids(&data_path, &self.schema, sorted_row_ids)
            }
        }
    }
}

/// Build HNSW index from a merged RecordBatch.
fn build_hnsw_index(batch: &RecordBatch, config: &CollectionConfig) -> Result<HnswIndex> {
    let index = HnswIndex::new(
        config.dimension,
        config.distance_metric,
        &config.index_config,
    );

    let vector_col = batch
        .column_by_name(COL_VECTOR)
        .ok_or_else(|| VdbError::InvalidSchema("missing vector column".to_string()))?;

    let fsl = vector_col
        .as_fixed_size_list_opt()
        .ok_or_else(|| VdbError::InvalidSchema("vector column is not FixedSizeList".to_string()))?;

    let values = fsl
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| VdbError::InvalidSchema("vector values are not Float32".to_string()))?;

    let dim = config.dimension;
    let n = batch.num_rows();
    let mut ids = Vec::with_capacity(n);
    let mut vecs = Vec::with_capacity(n);
    let raw = values.values();

    for i in 0..n {
        ids.push(i as u64);
        let start = i * dim;
        let end = start + dim;
        vecs.push(&raw[start..end]);
    }

    index.insert(&ids, &vecs)?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::SystemTime;

    use arrow::array::{
        BooleanArray, FixedSizeListArray, Float32Array, StringArray, TimestampMicrosecondArray,
    };
    use arrow::datatypes::{DataType, Field, TimeUnit};
    use vdb_common::config::CollectionConfig;
    use vdb_common::metrics::DistanceMetric;
    use vdb_index::traits::VectorIndex;

    fn test_config(fmt: StorageFormat) -> CollectionConfig {
        CollectionConfig {
            name: "test".to_string(),
            dimension: 3,
            distance_metric: DistanceMetric::L2,
            index_config: Default::default(),
            metadata_fields: vec![],
            storage_format: fmt,
        }
    }

    fn make_batch(n: usize, dim: usize) -> (SchemaRef, RecordBatch) {
        let ids: Vec<String> = (0..n).map(|i| format!("id_{i}")).collect();
        let id_array = Arc::new(StringArray::from(ids)) as Arc<dyn arrow::array::Array>;

        let flat: Vec<f32> = (0..n)
            .flat_map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();
        let values = Float32Array::from(flat);
        let vector_array = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                Arc::new(values),
                None,
            )
            .unwrap(),
        ) as Arc<dyn arrow::array::Array>;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        let ts_array =
            Arc::new(TimestampMicrosecondArray::from(vec![now; n])) as Arc<dyn arrow::array::Array>;
        let deleted_array =
            Arc::new(BooleanArray::from(vec![false; n])) as Arc<dyn arrow::array::Array>;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("_id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new(
                "_created_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("_deleted", DataType::Boolean, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, vector_array, ts_array, deleted_array],
        )
        .unwrap();

        (schema, batch)
    }

    #[test]
    fn test_segment_vortex_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(StorageFormat::Vortex);
        let (schema, batch) = make_batch(5, 3);

        let seg = Segment::write(
            dir.path(),
            "seg_v",
            schema.clone(),
            &[batch.clone()],
            &config,
        )
        .unwrap();
        assert_eq!(seg.num_rows, 5);
        assert_eq!(seg.format, StorageFormat::Vortex);

        // Read all
        let all = seg.read_all().unwrap();
        assert_eq!(all.num_rows(), 5);

        // Verify data
        let id_col = as_string_array(all.column_by_name(COL_ID).unwrap());
        assert_eq!(id_col.value(0), "id_0");
        assert_eq!(id_col.value(4), "id_4");
    }

    #[test]
    fn test_segment_vortex_load() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(StorageFormat::Vortex);
        let (schema, batch) = make_batch(5, 3);

        let seg = Segment::write(dir.path(), "seg_v2", schema.clone(), &[batch], &config).unwrap();
        drop(seg);

        // Load from disk
        let loaded = Segment::load(dir.path(), "seg_v2", &config).unwrap();
        assert_eq!(loaded.num_rows, 5);
        assert_eq!(loaded.format, StorageFormat::Vortex);

        let all = loaded.read_all().unwrap();
        assert_eq!(all.num_rows(), 5);
    }

    #[test]
    fn test_segment_vortex_read_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(StorageFormat::Vortex);
        let (schema, batch) = make_batch(10, 3);

        let seg =
            Segment::write(dir.path(), "seg_meta", schema.clone(), &[batch], &config).unwrap();

        let metadata = seg.read_metadata_by_row_ids(&[0, 3, 7]).unwrap();
        assert_eq!(metadata.num_rows(), 3);
        // Should not contain vector column
        assert!(metadata.column_by_name(COL_VECTOR).is_none());
        // Should contain _id column
        let id_col = as_string_array(metadata.column_by_name(COL_ID).unwrap());
        assert_eq!(id_col.value(0), "id_0");
        assert_eq!(id_col.value(1), "id_3");
        assert_eq!(id_col.value(2), "id_7");
    }

    #[test]
    fn test_segment_vortex_search() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(StorageFormat::Vortex);
        let (schema, batch) = make_batch(5, 3);

        let seg =
            Segment::write(dir.path(), "seg_search", schema.clone(), &[batch], &config).unwrap();

        // Search using the HNSW index
        let results = seg.index.search(&[1.0, 0.0, 0.0], 3, 50).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_segment_parquet_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(StorageFormat::Parquet);
        let (schema, batch) = make_batch(5, 3);

        let seg = Segment::write(
            dir.path(),
            "seg_pq",
            schema.clone(),
            &[batch.clone()],
            &config,
        )
        .unwrap();
        assert_eq!(seg.num_rows, 5);
        assert_eq!(seg.format, StorageFormat::Parquet);

        let all = seg.read_all().unwrap();
        assert_eq!(all.num_rows(), 5);

        let metadata = seg.read_metadata_by_row_ids(&[0, 2]).unwrap();
        assert_eq!(metadata.num_rows(), 2);
    }
}
