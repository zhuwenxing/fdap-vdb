use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use parking_lot::RwLock;

/// In-memory write buffer that accumulates RecordBatches.
pub struct MemTable {
    schema: SchemaRef,
    batches: RwLock<Vec<RecordBatch>>,
    row_count: RwLock<usize>,
}

impl MemTable {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            batches: RwLock::new(Vec::new()),
            row_count: RwLock::new(0),
        }
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn insert(&self, batch: RecordBatch) {
        let num_rows = batch.num_rows();
        self.batches.write().push(batch);
        *self.row_count.write() += num_rows;
    }

    pub fn row_count(&self) -> usize {
        *self.row_count.read()
    }

    pub fn should_flush(&self, max_rows: usize) -> bool {
        self.row_count() >= max_rows
    }

    /// Freeze and take all data, resetting the memtable.
    pub fn freeze(&self) -> Vec<RecordBatch> {
        let mut batches = self.batches.write();
        let mut count = self.row_count.write();
        *count = 0;
        std::mem::take(&mut *batches)
    }

    pub fn batches(&self) -> Vec<RecordBatch> {
        self.batches.read().clone()
    }

    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }
}
