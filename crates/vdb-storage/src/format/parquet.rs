use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow_schema::{Schema, SchemaRef};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use vdb_common::error::{Result, VdbError};
use vdb_common::schema::COL_VECTOR;

/// Write a RecordBatch to a Parquet file with ZSTD compression.
pub fn write_data(path: &Path, schema: SchemaRef, batch: &RecordBatch) -> Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(batch)?;
    writer
        .close()
        .map_err(|e| VdbError::Storage(e.to_string()))?;
    Ok(())
}

/// Load schema and total row count from a Parquet file.
pub fn load_schema_and_rows(path: &Path) -> Result<(SchemaRef, usize)> {
    let file = fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = reader.schema().clone();
    let reader = reader.build()?;
    let num_rows: usize = reader
        .into_iter()
        .filter_map(|b| b.ok())
        .map(|b| b.num_rows())
        .sum();
    Ok((schema, num_rows))
}

/// Read all data from a Parquet file.
pub fn read_all(path: &Path, schema: &SchemaRef) -> Result<RecordBatch> {
    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder
        .build()
        .map_err(|e| VdbError::Storage(e.to_string()))?;
    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches = batches.map_err(|e| VdbError::Storage(e.to_string()))?;
    concat_batches(schema, &batches).map_err(Into::into)
}

/// Read non-vector columns for specific rows using Parquet RowSelection.
///
/// `sorted_row_ids` must be sorted in ascending order.
pub fn read_metadata_by_row_ids(
    path: &Path,
    schema: &SchemaRef,
    sorted_row_ids: &[u64],
) -> Result<RecordBatch> {
    if sorted_row_ids.is_empty() {
        let projected = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .filter(|f| f.name() != COL_VECTOR)
                .cloned()
                .collect::<Vec<_>>(),
        ));
        return Ok(RecordBatch::new_empty(projected));
    }

    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let total_rows = builder.metadata().file_metadata().num_rows() as usize;

    // Column projection: exclude the vector column
    let arrow_schema = builder.schema();
    let proj_indices: Vec<usize> = arrow_schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name() != COL_VECTOR)
        .map(|(i, _)| i)
        .collect();
    let mask = ProjectionMask::roots(builder.parquet_schema(), proj_indices);

    // Row selection
    let selection = build_row_selection(sorted_row_ids, total_rows);

    let reader = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| VdbError::Storage(e.to_string()))?;

    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches = batches.map_err(|e| VdbError::Storage(e.to_string()))?;

    if batches.is_empty() {
        let projected = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .filter(|f| f.name() != COL_VECTOR)
                .cloned()
                .collect::<Vec<_>>(),
        ));
        return Ok(RecordBatch::new_empty(projected));
    }

    concat_batches(&batches[0].schema(), &batches).map_err(Into::into)
}

/// Build a RowSelection that picks only the given (sorted, ascending) row offsets.
fn build_row_selection(sorted_ids: &[u64], total_rows: usize) -> RowSelection {
    let mut selectors = Vec::with_capacity(sorted_ids.len() * 2);
    let mut cursor = 0usize;

    for &id in sorted_ids {
        let id = id as usize;
        if id >= total_rows {
            break;
        }
        if id > cursor {
            selectors.push(RowSelector::skip(id - cursor));
        }
        selectors.push(RowSelector::select(1));
        cursor = id + 1;
    }

    RowSelection::from(selectors)
}
