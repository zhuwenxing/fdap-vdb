use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "fdap-vdb", about = "FDAP-based vector database")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the vector database server
    Serve {
        /// Data directory for storage
        #[arg(long, default_value = "./data")]
        data_dir: String,

        /// Port to listen on
        #[arg(long, default_value_t = 50051)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("fdap_vdb=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { data_dir, port } => {
            let config = vdb_server::ServerConfig { data_dir, port };
            vdb_server::run_server(config).await?;
        }
    }

    Ok(())
}
