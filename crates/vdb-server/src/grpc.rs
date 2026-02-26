use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use vdb_common::config::{CollectionConfig, MetadataFieldConfig, MetadataFieldType};
use vdb_common::metrics::DistanceMetric;
use vdb_storage::engine::{MetadataColumnValue, StorageEngine};

use crate::proto;

pub struct VdbGrpcService {
    engine: Arc<StorageEngine>,
}

impl VdbGrpcService {
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl proto::vdb_service_server::VdbService for VdbGrpcService {
    async fn create_collection(
        &self,
        request: Request<proto::CreateCollectionRequest>,
    ) -> Result<Response<proto::CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        let metric = if req.distance_metric.is_empty() {
            DistanceMetric::Cosine
        } else {
            req.distance_metric
                .parse::<DistanceMetric>()
                .map_err(Status::invalid_argument)?
        };

        let metadata_fields: Vec<MetadataFieldConfig> = req
            .metadata_fields
            .iter()
            .map(|f| {
                let field_type = f
                    .field_type
                    .parse::<MetadataFieldType>()
                    .map_err(Status::invalid_argument)?;
                Ok(MetadataFieldConfig {
                    name: f.name.clone(),
                    field_type,
                })
            })
            .collect::<Result<_, Status>>()?;

        let config = CollectionConfig {
            name: req.name,
            dimension: req.dimension as usize,
            distance_metric: metric,
            index_config: Default::default(),
            metadata_fields,
        };

        self.engine
            .create_collection(config)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::CreateCollectionResponse {}))
    }

    async fn list_collections(
        &self,
        _request: Request<proto::ListCollectionsRequest>,
    ) -> Result<Response<proto::ListCollectionsResponse>, Status> {
        let configs = self.engine.list_collections();
        let collections = configs
            .into_iter()
            .map(|c| proto::CollectionInfo {
                name: c.name,
                dimension: c.dimension as u32,
                distance_metric: c.distance_metric.as_str().to_string(),
            })
            .collect();

        Ok(Response::new(proto::ListCollectionsResponse {
            collections,
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<proto::DropCollectionRequest>,
    ) -> Result<Response<proto::DropCollectionResponse>, Status> {
        let req = request.into_inner();
        self.engine
            .drop_collection(&req.name)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::DropCollectionResponse {}))
    }

    async fn insert(
        &self,
        request: Request<proto::InsertRequest>,
    ) -> Result<Response<proto::InsertResponse>, Status> {
        let req = request.into_inner();

        let ids = req.ids;
        let vectors: Vec<Vec<f32>> = req.vectors.into_iter().map(|v| v.values).collect();

        let mut metadata: HashMap<String, Vec<MetadataColumnValue>> = HashMap::new();
        for (key, mv) in req.metadata {
            let values: Vec<MetadataColumnValue> = mv
                .values
                .into_iter()
                .map(|v| match v.value {
                    Some(proto::metadata_value::Value::StringValue(s)) => {
                        MetadataColumnValue::String(s)
                    }
                    Some(proto::metadata_value::Value::IntValue(i)) => {
                        MetadataColumnValue::Int64(i)
                    }
                    Some(proto::metadata_value::Value::FloatValue(f)) => {
                        MetadataColumnValue::Float64(f)
                    }
                    Some(proto::metadata_value::Value::BoolValue(b)) => {
                        MetadataColumnValue::Bool(b)
                    }
                    None => MetadataColumnValue::Null,
                })
                .collect();
            metadata.insert(key, values);
        }

        let count = self
            .engine
            .insert(&req.collection, ids, vectors, metadata)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Auto-flush check
        let _ = self.engine.maybe_flush(&req.collection, 100_000);

        Ok(Response::new(proto::InsertResponse { count }))
    }

    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let req = request.into_inner();

        let query = req
            .query_vector
            .ok_or_else(|| Status::invalid_argument("missing query_vector"))?
            .values;
        let top_k = req.top_k as usize;

        let hits = self
            .engine
            .search(&req.collection, &query, top_k, 0)
            .map_err(|e| Status::internal(e.to_string()))?;

        let results = hits
            .into_iter()
            .map(|h| proto::SearchResult {
                id: h.id,
                distance: h.distance,
                metadata: h.metadata,
            })
            .collect();

        Ok(Response::new(proto::SearchResponse { results }))
    }

    async fn delete(
        &self,
        request: Request<proto::DeleteRequest>,
    ) -> Result<Response<proto::DeleteResponse>, Status> {
        let req = request.into_inner();
        let count = self
            .engine
            .delete(&req.collection, &req.ids)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::DeleteResponse { count }))
    }

    async fn compact(
        &self,
        request: Request<proto::CompactRequest>,
    ) -> Result<Response<proto::CompactResponse>, Status> {
        let req = request.into_inner();
        let (before, after, removed) = self
            .engine
            .compact(&req.collection)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::CompactResponse {
            segments_before: before as u64,
            segments_after: after as u64,
            rows_removed: removed as u64,
        }))
    }

    async fn flush(
        &self,
        request: Request<proto::FlushRequest>,
    ) -> Result<Response<proto::FlushResponse>, Status> {
        let req = request.into_inner();
        self.engine
            .flush(&req.collection)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(proto::FlushResponse {}))
    }
}
