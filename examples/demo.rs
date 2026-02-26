/// fdap-vdb 完整使用示例
///
/// 启动 gRPC + FlightSQL 服务器，然后通过客户端 SDK 演示：
///   1. 创建集合（指定维度、距离度量、metadata schema）
///   2. 批量插入向量 + 元数据
///   3. 相似度搜索（Top-K）
///   4. Flush 持久化到 Parquet + HNSW 索引
///   5. Flush 后再次搜索（跨 segment + memtable）
///   6. 通过 FlightSQL 执行 SQL 查询
///   7. Delete 软删除 + 验证搜索结果
///   8. Compact 合并 segment + 物理清理
///   9. 列出 / 删除集合
///
/// 运行: cargo run --example demo
use std::collections::HashMap;
use std::time::Duration;

use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_flight::FlightInfo;
use futures::TryStreamExt;
use tracing_subscriber::EnvFilter;
use vdb_client::client::{MetadataColumnValue, VdbClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("fdap_vdb=info".parse().unwrap()),
        )
        .init();

    // ── 启动服务器 ────────────────────────────────────────────────
    let data_dir = tempfile::tempdir()?;
    let port = 50052u16;
    let server_config = vdb_server::ServerConfig {
        data_dir: data_dir.path().to_string_lossy().to_string(),
        port,
    };

    tokio::spawn(async move {
        if let Err(e) = vdb_server::run_server(server_config).await {
            eprintln!("server error: {e}");
        }
    });

    // 等待服务器就绪（带重试）
    let addr = format!("http://localhost:{port}");
    let mut client = loop {
        match VdbClient::connect(&addr).await {
            Ok(c) => break c,
            Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    };
    println!("connected to server at {addr}\n");

    // ── 1. 创建集合 ──────────────────────────────────────────────
    println!("=== Step 1: Create collection ===");
    client
        .create_collection(
            "articles",
            4,
            "cosine",
            vec![("category", "string"), ("score", "int64")],
        )
        .await?;
    println!("  created collection 'articles' (dim=4, metric=cosine)");
    println!("  metadata fields: category(string), score(int64)\n");

    // ── 2. 插入向量 ──────────────────────────────────────────────
    println!("=== Step 2: Insert vectors ===");
    let n = 20;
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut categories = Vec::new();
    let mut scores = Vec::new();

    for i in 0..n {
        ids.push(format!("article_{i}"));
        // 在单位圆上均匀分布，让搜索结果有明确的距离差异
        let angle = (i as f32) * std::f32::consts::PI * 2.0 / (n as f32);
        vectors.push(vec![angle.cos(), angle.sin(), 0.0, 0.0]);
        let cat = match i % 3 {
            0 => "science",
            1 => "tech",
            _ => "art",
        };
        categories.push(MetadataColumnValue::String(cat.to_string()));
        scores.push(MetadataColumnValue::Int64((i * 10) as i64));
    }

    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), categories);
    metadata.insert("score".to_string(), scores);

    let count = client
        .insert("articles", ids, vectors, metadata)
        .await?;
    println!("  inserted {count} vectors\n");

    // ── 3. 搜索（数据在 memtable 中，暴力扫描） ─────────────────
    println!("=== Step 3: Search (memtable, brute-force) ===");
    println!("  query vector: [1.0, 0.0, 0.0, 0.0]  (same direction as article_0)");
    let hits = client
        .search("articles", vec![1.0, 0.0, 0.0, 0.0], 5, "cosine", "")
        .await?;
    print_hits(&hits);

    // ── 4. Flush 持久化 ──────────────────────────────────────────
    println!("=== Step 4: Flush to Parquet + HNSW index ===");
    client.flush("articles").await?;
    println!("  flushed to disk\n");

    // ── 5. 插入更多数据，搜索跨 segment + memtable ──────────────
    println!("=== Step 5: Insert more + search across segment & memtable ===");
    let extra_ids = vec!["extra_near".to_string(), "extra_far".to_string()];
    let extra_vecs = vec![
        vec![0.99, 0.01, 0.0, 0.0], // 非常接近 query
        vec![-1.0, 0.0, 0.0, 0.0],  // 完全相反
    ];
    let mut extra_meta = HashMap::new();
    extra_meta.insert(
        "category".to_string(),
        vec![
            MetadataColumnValue::String("science".to_string()),
            MetadataColumnValue::String("art".to_string()),
        ],
    );
    extra_meta.insert(
        "score".to_string(),
        vec![
            MetadataColumnValue::Int64(999),
            MetadataColumnValue::Int64(0),
        ],
    );
    client
        .insert("articles", extra_ids, extra_vecs, extra_meta)
        .await?;
    println!("  inserted 2 more vectors (extra_near, extra_far)");

    let hits = client
        .search("articles", vec![1.0, 0.0, 0.0, 0.0], 5, "cosine", "")
        .await?;
    println!("  search results (segment + memtable combined):");
    print_hits(&hits);

    // ── 6. FlightSQL: 用 SQL 查询 ───────────────────────────────
    println!("=== Step 6: FlightSQL - SQL query ===");
    let sql = "SELECT _id, category, score FROM articles LIMIT 10";
    println!("  SQL: {sql}");

    let mut flight_client = FlightSqlServiceClient::new(
        tonic::transport::Channel::from_shared(addr.clone())?
            .connect()
            .await?,
    );
    flight_client.handshake("", "").await.ok(); // optional handshake

    let info: FlightInfo = flight_client.execute(sql.to_string(), None).await?;
    let ticket = info.endpoint[0]
        .ticket
        .as_ref()
        .expect("missing ticket")
        .clone();
    let batch_stream = flight_client.do_get(ticket).await?;
    let batches: Vec<_> = batch_stream.try_collect().await?;

    println!("  result ({} batches):", batches.len());
    for batch in &batches {
        println!("{}", arrow::util::pretty::pretty_format_batches(&[batch.clone()])?);
    }
    println!();

    // ── 7. Delete 软删除 ──────────────────────────────────────────
    println!("=== Step 7: Delete vectors ===");
    let del_count = client
        .delete("articles", vec!["article_0".to_string(), "extra_near".to_string()])
        .await?;
    println!("  deleted {del_count} vectors (article_0, extra_near)");

    let hits = client
        .search("articles", vec![1.0, 0.0, 0.0, 0.0], 5, "cosine", "")
        .await?;
    println!("  search after delete (article_0 and extra_near should be gone):");
    print_hits(&hits);

    // ── 8. Compact 合并 + 物理清理 ──────────────────────────────
    println!("=== Step 8: Compact segments ===");
    // Flush extra data first to create a second segment
    client.flush("articles").await?;
    println!("  flushed extra data to create 2nd segment");

    let (before, after, removed) = client.compact("articles").await?;
    println!("  compact result: {before} segments → {after} segment, {removed} rows removed");

    let hits = client
        .search("articles", vec![1.0, 0.0, 0.0, 0.0], 5, "cosine", "")
        .await?;
    println!("  search after compact:");
    print_hits(&hits);

    // ── 9. 列出 & 删除集合 ───────────────────────────────────────
    println!("=== Step 9: List & drop collection ===");
    let collections = client.list_collections().await?;
    for col in &collections {
        println!(
            "  collection: {} (dim={}, metric={})",
            col.name, col.dimension, col.distance_metric
        );
    }

    // 创建第二个集合，然后删除
    client
        .create_collection("temp_collection", 128, "l2", vec![])
        .await?;
    println!("  created 'temp_collection'");

    let collections = client.list_collections().await?;
    println!("  collections count: {}", collections.len());

    client.drop_collection("temp_collection").await?;
    println!("  dropped 'temp_collection'");
    let collections = client.list_collections().await?;
    println!("  remaining collections: {}", collections.len());

    println!("\n=== Demo complete! ===");
    println!("  data directory: {}", data_dir.path().display());
    // 列出生成的文件
    print_tree(data_dir.path(), 0);

    Ok(())
}

fn print_hits(hits: &[vdb_client::client::SearchHit]) {
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "  #{}: id={:<15} distance={:.6}  metadata={:?}",
            i + 1,
            hit.id,
            hit.distance,
            hit.metadata
        );
    }
    println!();
}

fn print_tree(path: &std::path::Path, depth: usize) {
    let indent = "  ".repeat(depth + 1);
    if let Ok(entries) = std::fs::read_dir(path) {
        let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if entry.path().is_dir() {
                println!("{indent}{name}/");
                print_tree(&entry.path(), depth + 1);
            } else {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                println!("{indent}{name} ({size} bytes)");
            }
        }
    }
}
