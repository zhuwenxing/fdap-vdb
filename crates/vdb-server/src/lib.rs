pub mod flight;
pub mod grpc;

#[allow(clippy::all)]
pub mod proto {
    tonic::include_proto!("vdb.v1");
}

use std::path::Path;
use std::sync::Arc;

use vdb_common::config::CompactionConfig;
use vdb_common::error::Result;
use vdb_storage::engine::StorageEngine;

pub struct ServerConfig {
    pub data_dir: String,
    pub port: u16,
    pub compaction: CompactionConfig,
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
    let flight_svc = flight::flight_service_server(engine.clone());

    // Start background compaction task
    spawn_compaction_task(engine, config.compaction);

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

fn spawn_compaction_task(engine: Arc<StorageEngine>, config: CompactionConfig) {
    tokio::spawn(async move {
        let interval = tokio::time::Duration::from_secs(config.check_interval_secs);
        tracing::info!(
            "background compaction started: check every {}s, max_segments={}, delete_ratio={:.0}%",
            config.check_interval_secs,
            config.max_segments,
            config.delete_ratio_threshold * 100.0,
        );

        loop {
            tokio::time::sleep(interval).await;

            let collections = engine.list_collections();
            for col_config in &collections {
                let name = &col_config.name;
                let stats = match engine.compaction_stats(name) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let (seg_count, total_rows, deleted_count) = stats;
                if seg_count < 2 && deleted_count == 0 {
                    continue;
                }

                let too_many_segments = seg_count >= config.max_segments;
                let high_delete_ratio = total_rows > 0
                    && (deleted_count as f64 / total_rows as f64) > config.delete_ratio_threshold;

                if too_many_segments || high_delete_ratio {
                    let reason = if too_many_segments && high_delete_ratio {
                        format!("{seg_count} segments + {deleted_count}/{total_rows} deleted")
                    } else if too_many_segments {
                        format!("{seg_count} segments (threshold: {})", config.max_segments)
                    } else {
                        format!(
                            "{deleted_count}/{total_rows} deleted ({:.0}%)",
                            deleted_count as f64 / total_rows as f64 * 100.0
                        )
                    };

                    tracing::info!("auto-compact triggered for '{name}': {reason}");
                    match engine.compact(name) {
                        Ok((before, after, removed)) => {
                            tracing::info!(
                                "auto-compact '{name}' done: {before} → {after} segments, {removed} rows removed"
                            );
                        }
                        Err(e) => {
                            tracing::warn!("auto-compact '{name}' failed: {e}");
                        }
                    }
                }
            }
        }
    });
}
