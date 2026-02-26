use std::pin::Pin;
use std::sync::Arc;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{CommandStatementQuery, TicketStatementQuery};
use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
};
use datafusion::prelude::SessionContext;
use futures::{Stream, TryStreamExt};
use prost::Message;
use tonic::{Request, Response, Status, Streaming};
use vdb_query::context::create_session_context;
use vdb_storage::engine::StorageEngine;

#[derive(Clone)]
pub struct VdbFlightSqlService {
    engine: Arc<StorageEngine>,
}

impl VdbFlightSqlService {
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self { engine }
    }

    fn session_context(&self) -> Result<SessionContext, Status> {
        create_session_context(self.engine.clone()).map_err(|e| Status::internal(e.to_string()))
    }
}

#[tonic::async_trait]
impl FlightSqlService for VdbFlightSqlService {
    type FlightService = VdbFlightSqlService;

    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<
        Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>,
        Status,
    > {
        let result = HandshakeResponse {
            protocol_version: 0,
            payload: bytes::Bytes::new(),
        };
        let stream = futures::stream::once(async { Ok(result) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let ctx = self.session_context()?;
        let df = ctx
            .sql(&query.query)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let schema = df.schema().inner().clone();

        // Encode the query string as the ticket handle
        let ticket_query = TicketStatementQuery {
            statement_handle: query.query.clone().into(),
        };
        let ticket = Ticket::new(ticket_query.encode_to_vec());

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(FlightEndpoint::new().with_ticket(ticket));

        Ok(Response::new(info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let query = String::from_utf8(ticket.statement_handle.to_vec())
            .map_err(|e| Status::internal(e.to_string()))?;

        let ctx = self.session_context()?;
        let df = ctx
            .sql(&query)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let schema = df.schema().inner().clone();
        let batches = df
            .collect()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let batch_stream = futures::stream::iter(batches.into_iter().map(Ok));
        let flight_data_stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(batch_stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(flight_data_stream)))
    }

    async fn register_sql_info(&self, _id: i32, _result: &arrow_flight::sql::SqlInfo) {}
}

pub fn flight_service_server(
    engine: Arc<StorageEngine>,
) -> FlightServiceServer<impl FlightService> {
    let service = VdbFlightSqlService::new(engine);
    FlightServiceServer::new(service)
}
