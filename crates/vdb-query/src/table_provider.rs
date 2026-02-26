use std::any::Any;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::Result as DfResult;
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::logical_expr::TableType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use vdb_storage::engine::StorageEngine;

/// DataFusion TableProvider wrapping a vector collection.
pub struct CollectionTableProvider {
    collection_name: String,
    schema: SchemaRef,
    engine: Arc<StorageEngine>,
}

impl fmt::Debug for CollectionTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CollectionTableProvider")
            .field("collection_name", &self.collection_name)
            .finish()
    }
}

impl CollectionTableProvider {
    pub fn new(collection_name: String, schema: SchemaRef, engine: Arc<StorageEngine>) -> Self {
        Self {
            collection_name,
            schema,
            engine,
        }
    }
}

#[async_trait]
impl TableProvider for CollectionTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let batches = self
            .engine
            .get_collection_data(&self.collection_name)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let provider = MemTable::try_new(self.schema.clone(), vec![batches])?;
        provider.scan(state, projection, &[], None).await
    }
}
