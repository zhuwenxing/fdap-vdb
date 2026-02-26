use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{as_string_array, AsArray, BooleanArray, Float32Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use vdb_common::config::CollectionConfig;
use vdb_common::error::{Result, VdbError};
use vdb_common::schema::{COL_DELETED, COL_ID, COL_VECTOR};
use vdb_index::hnsw::HnswIndex;
use vdb_index::traits::VectorIndex;

pub struct Segment {
    pub id: String,
    path: PathBuf,
    pub index: Arc<HnswIndex>,
    pub schema: SchemaRef,
    pub num_rows: usize,
}

impl Segment {
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

        // Write Parquet
        let parquet_path = seg_dir.join("data.parquet");
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        let file = fs::File::create(&parquet_path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        writer.write(&merged)?;
        writer
            .close()
            .map_err(|e| VdbError::Storage(e.to_string()))?;

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
        })
    }

    /// Load an existing segment from disk.
    pub fn load(base: &Path, seg_id: &str, config: &CollectionConfig) -> Result<Self> {
        let seg_dir = base.join(seg_id);
        let parquet_path = seg_dir.join("data.parquet");
        let index_dir = seg_dir.join("hnsw_index");

        // Read parquet to get schema and row count
        let file = fs::File::open(&parquet_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = reader.schema().clone();
        let reader = reader.build()?;
        let num_rows: usize = reader
            .into_iter()
            .filter_map(|b| b.ok())
            .map(|b| b.num_rows())
            .sum();

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
        })
    }

    /// Read all data from the parquet file.
    pub fn read_all(&self) -> Result<RecordBatch> {
        let path = self.path.join("data.parquet");
        let file = fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder
            .build()
            .map_err(|e| VdbError::Storage(e.to_string()))?;
        let batches: std::result::Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(|e| VdbError::Storage(e.to_string()))?;
        concat_batches(&self.schema, &batches).map_err(Into::into)
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
