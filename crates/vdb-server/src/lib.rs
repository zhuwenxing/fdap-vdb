pub mod flight;
pub mod grpc;

#[allow(clippy::all)]
pub mod proto {
    tonic::include_proto!("vdb.v1");
}

use std::path::Path;
use std::sync::Arc;

use vdb_common::error::Result;
use vdb_storage::engine::StorageEngine;

pub struct ServerConfig {
    pub data_dir: String,
    pub port: u16,
}

pub async fn run_server(config: ServerConfig) -> Result<()> {
    let engine = Arc::new(StorageEngine::open(Path::new(&config.data_dir))?);

    let addr =
        format!("0.0.0.0:{}", config.port)
            .parse()
            .map_err(|e: std::net::AddrParseError| {
                vdb_common::error::VdbError::Internal(e.to_string())
            })?;

    let grpc_service = grpc::VdbGrpcService::new(engine.clone());
    let flight_svc = flight::flight_service_server(engine);

    tracing::info!("fdap-vdb server listening on {addr}");

    tonic::transport::Server::builder()
        .add_service(proto::vdb_service_server::VdbServiceServer::new(
            grpc_service,
        ))
        .add_service(flight_svc)
        .serve(addr)
        .await
        .map_err(|e| vdb_common::error::VdbError::Internal(e.to_string()))?;

    Ok(())
}
