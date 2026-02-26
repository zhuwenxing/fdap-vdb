use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use vdb_common::error::{Result, VdbError};

/// Write-Ahead Log using Arrow IPC stream format.
pub struct Wal {
    dir: PathBuf,
    collection: String,
}

impl Wal {
    pub fn new(data_dir: &Path, collection: &str) -> Result<Self> {
        let dir = data_dir.join("wal").join(collection);
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            collection: collection.to_string(),
        })
    }

    fn wal_path(&self) -> PathBuf {
        self.dir.join(format!("{}.wal", self.collection))
    }

    /// Append a RecordBatch to the WAL.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let path = self.wal_path();
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
            writer.write(batch)?;
            writer.finish()?;
        }

        let len = (buf.len() as u64).to_le_bytes();
        file.write_all(&len)?;
        file.write_all(&buf)?;
        file.sync_all()?;
        Ok(())
    }

    /// Read all batches from the WAL for recovery.
    pub fn read_all(&self, _schema: &SchemaRef) -> Result<Vec<RecordBatch>> {
        let path = self.wal_path();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let data = fs::read(&path)?;
        let mut offset = 0;
        let mut batches = Vec::new();

        while offset + 8 <= data.len() {
            let len_bytes: [u8; 8] = data[offset..offset + 8]
                .try_into()
                .map_err(|_| VdbError::Storage("corrupt WAL: invalid length".to_string()))?;
            let len = u64::from_le_bytes(len_bytes) as usize;
            offset += 8;

            if offset + len > data.len() {
                tracing::warn!("truncated WAL entry, skipping");
                break;
            }

            let ipc_data = &data[offset..offset + len];
            let cursor = std::io::Cursor::new(ipc_data);
            let reader = StreamReader::try_new(cursor, None)?;
            for batch_result in reader {
                batches.push(batch_result?);
            }
            offset += len;
        }

        Ok(batches)
    }

    /// Clear the WAL after successful flush.
    pub fn clear(&self) -> Result<()> {
        let path = self.wal_path();
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BooleanArray, FixedSizeListArray, Float32Array, StringArray, TimestampMicrosecondArray,
    };
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;

    fn make_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                false,
            ),
            Field::new(
                "_created_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("_deleted", DataType::Boolean, false),
        ]));

        let ids = StringArray::from(vec!["a", "b"]);
        let values = Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            3,
            Arc::new(values),
            None,
        )
        .unwrap();
        let ts = TimestampMicrosecondArray::from(vec![1000, 2000]);
        let deleted = BooleanArray::from(vec![false, false]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(vectors),
                Arc::new(ts),
                Arc::new(deleted),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_wal_append_read() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::new(dir.path(), "test").unwrap();
        let batch = make_test_batch();

        wal.append(&batch).unwrap();
        wal.append(&batch).unwrap();

        let schema = batch.schema();
        let batches = wal.read_all(&schema).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 2);

        wal.clear().unwrap();
        let batches = wal.read_all(&schema).unwrap();
        assert!(batches.is_empty());
    }
}
