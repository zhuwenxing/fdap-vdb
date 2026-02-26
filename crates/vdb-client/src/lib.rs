pub mod client;

#[allow(clippy::all)]
pub mod proto {
    tonic::include_proto!("vdb.v1");
}
