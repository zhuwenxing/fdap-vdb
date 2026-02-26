use std::sync::Arc;

use datafusion::prelude::SessionContext;
use vdb_common::error::{Result, VdbError};
use vdb_storage::engine::StorageEngine;

use crate::table_provider::CollectionTableProvider;
use crate::udf::distance::{cosine_distance_udf, inner_product_distance_udf, l2_distance_udf};

/// Create a DataFusion SessionContext with registered collections and UDFs.
pub fn create_session_context(engine: Arc<StorageEngine>) -> Result<SessionContext> {
    let ctx = SessionContext::new();

    ctx.register_udf(cosine_distance_udf());
    ctx.register_udf(l2_distance_udf());
    ctx.register_udf(inner_product_distance_udf());

    for config in engine.list_collections() {
        let schema = engine.get_collection_schema(&config.name)?;
        let provider = CollectionTableProvider::new(config.name.clone(), schema, engine.clone());
        ctx.register_table(&config.name, Arc::new(provider))
            .map_err(|e| VdbError::Query(e.to_string()))?;
    }

    Ok(ctx)
}

/// Register a single collection into an existing context.
pub fn register_collection(
    ctx: &SessionContext,
    engine: Arc<StorageEngine>,
    collection_name: &str,
) -> Result<()> {
    let schema = engine.get_collection_schema(collection_name)?;
    let provider =
        CollectionTableProvider::new(collection_name.to_string(), schema, engine.clone());
    ctx.register_table(collection_name, Arc::new(provider))
        .map_err(|e| VdbError::Query(e.to_string()))?;
    Ok(())
}
