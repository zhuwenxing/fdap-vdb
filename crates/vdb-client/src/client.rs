use std::collections::HashMap;

use tonic::transport::Channel;

use crate::proto::{
    self, vdb_service_client::VdbServiceClient, CreateCollectionRequest, FieldSchema,
    InsertRequest, ListCollectionsRequest, MetadataValue, MetadataValues, SearchRequest, Vector,
};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("grpc error: {0}")]
    Grpc(#[from] tonic::Status),
}

pub type Result<T> = std::result::Result<T, ClientError>;

pub struct VdbClient {
    inner: VdbServiceClient<Channel>,
}

impl VdbClient {
    pub async fn connect(url: &str) -> Result<Self> {
        let inner = VdbServiceClient::connect(url.to_string()).await?;
        Ok(Self { inner })
    }

    pub async fn create_collection(
        &mut self,
        name: &str,
        dimension: u32,
        distance_metric: &str,
        metadata_fields: Vec<(&str, &str)>,
    ) -> Result<()> {
        let fields = metadata_fields
            .into_iter()
            .map(|(n, t)| FieldSchema {
                name: n.to_string(),
                field_type: t.to_string(),
            })
            .collect();

        self.inner
            .create_collection(CreateCollectionRequest {
                name: name.to_string(),
                dimension,
                distance_metric: distance_metric.to_string(),
                metadata_fields: fields,
            })
            .await?;
        Ok(())
    }

    pub async fn list_collections(&mut self) -> Result<Vec<proto::CollectionInfo>> {
        let resp = self
            .inner
            .list_collections(ListCollectionsRequest {})
            .await?;
        Ok(resp.into_inner().collections)
    }

    pub async fn insert(
        &mut self,
        collection: &str,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: HashMap<String, Vec<MetadataColumnValue>>,
    ) -> Result<u64> {
        let proto_vectors = vectors.into_iter().map(|v| Vector { values: v }).collect();

        let proto_metadata: HashMap<String, MetadataValues> = metadata
            .into_iter()
            .map(|(key, vals)| {
                let proto_vals = vals
                    .into_iter()
                    .map(|v| match v {
                        MetadataColumnValue::String(s) => MetadataValue {
                            value: Some(proto::metadata_value::Value::StringValue(s)),
                        },
                        MetadataColumnValue::Int64(i) => MetadataValue {
                            value: Some(proto::metadata_value::Value::IntValue(i)),
                        },
                        MetadataColumnValue::Float64(f) => MetadataValue {
                            value: Some(proto::metadata_value::Value::FloatValue(f)),
                        },
                        MetadataColumnValue::Bool(b) => MetadataValue {
                            value: Some(proto::metadata_value::Value::BoolValue(b)),
                        },
                        MetadataColumnValue::Null => MetadataValue { value: None },
                    })
                    .collect();
                (key, MetadataValues { values: proto_vals })
            })
            .collect();

        let resp = self
            .inner
            .insert(InsertRequest {
                collection: collection.to_string(),
                ids,
                vectors: proto_vectors,
                metadata: proto_metadata,
            })
            .await?;
        Ok(resp.into_inner().count)
    }

    pub async fn search(
        &mut self,
        collection: &str,
        query_vector: Vec<f32>,
        top_k: u32,
        distance_metric: &str,
        filter: &str,
    ) -> Result<Vec<SearchHit>> {
        let resp = self
            .inner
            .search(SearchRequest {
                collection: collection.to_string(),
                query_vector: Some(Vector {
                    values: query_vector,
                }),
                top_k,
                distance_metric: distance_metric.to_string(),
                filter: filter.to_string(),
            })
            .await?;

        let hits = resp
            .into_inner()
            .results
            .into_iter()
            .map(|r| SearchHit {
                id: r.id,
                distance: r.distance,
                metadata: r.metadata,
            })
            .collect();
        Ok(hits)
    }
}

#[derive(Debug, Clone)]
pub enum MetadataColumnValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
}
