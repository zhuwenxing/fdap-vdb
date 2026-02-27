use std::path::Path;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow_schema::{Schema, SchemaRef};
use futures::TryStreamExt;
use vdb_common::error::{Result, VdbError};
use vdb_common::schema::COL_VECTOR;
use vortex::array::arrow::FromArrowArray;
use vortex::array::ArrayRef;
use vortex::expr::{root, select_exclude};
use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

/// Dedicated tokio runtime for Vortex async operations.
/// Uses a separate runtime to avoid nesting block_on inside a tokio worker thread.
static VORTEX_RT: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to create Vortex runtime")
});

/// Create a VortexSession. Must be called from within VORTEX_RT context
/// so that Handle::find() can locate the tokio runtime.
fn session() -> VortexSession {
    VortexSession::default()
}

/// Write a RecordBatch to a Vortex file.
pub fn write_data(path: &Path, _schema: SchemaRef, batch: &RecordBatch) -> Result<()> {
    let vortex_array = ArrayRef::from_arrow(batch.clone(), false).map_err(map_vortex_err)?;

    // Write into a Vec<u8> buffer (VortexWrite is implemented for Vec<u8>),
    // then flush to disk synchronously.
    let buf = VORTEX_RT.block_on(async {
        let session = session();
        let mut buf = Vec::new();
        session
            .write_options()
            .write(&mut buf, vortex_array.to_array_stream())
            .await
            .map_err(map_vortex_err)?;
        Ok::<_, VdbError>(buf)
    })?;

    std::fs::write(path, &buf)?;
    Ok(())
}

/// Get the row count from a Vortex file.
pub fn row_count(path: &Path) -> Result<u64> {
    VORTEX_RT.block_on(async {
        let session = session();
        let vf = session
            .open_options()
            .open_path(path)
            .await
            .map_err(map_vortex_err)?;
        Ok(vf.row_count())
    })
}

/// Read all data from a Vortex file as a RecordBatch.
pub fn read_all(path: &Path, schema: &SchemaRef) -> Result<RecordBatch> {
    let schema = schema.clone();
    VORTEX_RT.block_on(async {
        let session = session();
        let vf = session
            .open_options()
            .open_path(path)
            .await
            .map_err(map_vortex_err)?;

        let stream = vf
            .scan()
            .map_err(map_vortex_err)?
            .into_record_batch_stream(schema.clone())
            .map_err(map_vortex_err)?;

        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(map_vortex_err)?;

        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        concat_batches(&schema, &batches).map_err(Into::into)
    })
}

/// Read non-vector columns for specific rows from a Vortex file.
///
/// `sorted_row_ids` must be sorted in ascending order.
pub fn read_metadata_by_row_ids(
    path: &Path,
    schema: &SchemaRef,
    sorted_row_ids: &[u64],
) -> Result<RecordBatch> {
    let projected_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .filter(|f| f.name() != COL_VECTOR)
            .cloned()
            .collect::<Vec<_>>(),
    ));

    if sorted_row_ids.is_empty() {
        return Ok(RecordBatch::new_empty(projected_schema));
    }

    let sorted_row_ids = sorted_row_ids.to_vec();
    VORTEX_RT.block_on(async {
        let session = session();
        let vf = session
            .open_options()
            .open_path(path)
            .await
            .map_err(map_vortex_err)?;

        // Read with column projection (exclude vector), then filter to needed rows.
        let stream = vf
            .scan()
            .map_err(map_vortex_err)?
            .with_projection(select_exclude([COL_VECTOR], root()))
            .into_record_batch_stream(projected_schema.clone())
            .map_err(map_vortex_err)?;

        let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(map_vortex_err)?;

        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(projected_schema));
        }

        let all = concat_batches(&projected_schema, &batches)?;

        // Select specific rows by their row offsets
        let indices: Vec<u64> = sorted_row_ids
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

        RecordBatch::try_new(projected_schema, columns).map_err(Into::into)
    })
}

fn map_vortex_err(e: impl std::fmt::Display) -> VdbError {
    VdbError::Vortex(e.to_string())
}
