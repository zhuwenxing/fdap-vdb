# fdap-vdb

基于 **FDAP**（Flight + DataFusion + Arrow + Parquet/Vortex）技术栈构建的单机向量数据库，面向 RAG / 语义搜索场景。

## 特性

- **HNSW 向量索引** — 基于 `hnsw_rs`，支持 Cosine / L2 / Inner Product 三种距离度量
- **Arrow 原生存储** — 向量以 `FixedSizeList<Float32>` 存储，元数据以 Arrow 列式格式管理
- **双存储格式** — 支持 Vortex (默认) 和 Parquet 两种列式格式，可按集合切换
- **WAL 保障** — Arrow IPC 格式预写日志，crash recovery
- **DataFusion SQL** — 每个集合注册为 DataFusion 表，支持向量距离 UDF + SQL 查询
- **双协议服务** — gRPC 自定义接口 + FlightSQL 标准协议，同一端口
- **软删除 + 自动 Compact** — 按 ID 软删除，后台自动合并 Segment 并物理清理
- **Rust 客户端 SDK** — 类型安全的 gRPC 客户端

## 架构

```
Client (gRPC / FlightSQL)
         |
    vdb-server          ← tonic + arrow-flight
         |
    vdb-query           ← DataFusion TableProvider + 距离 UDF
         |
    vdb-storage         ← WAL → MemTable → Vortex/Parquet Segment + HNSW Index
         |
    vdb-index           ← hnsw_rs
         |
    vdb-common          ← 共享类型、配置、Schema
```

**写入流程**: Client → WAL 追加 → MemTable 缓冲 → 触发阈值 → Flush → Vortex/Parquet + HNSW 索引 → 清理 WAL

**搜索流程**: Query → 各 Segment HNSW Top-K → MemTable 暴力扫描 → 合并排序 → 返回 Top-K

**删除流程**: Delete(IDs) → 记录到 deleted_ids 集合 → 持久化到 JSON → 搜索时自动过滤

**Compact 流程**: Flush MemTable → 读取所有 Segment → 过滤已删除行 → 写入单个新 Segment → 清理旧文件

## 快速开始

### 启动服务器

```bash
cargo run -- serve --data-dir ./data --port 50051
```

### 运行完整示例

```bash
cargo run --example demo
```

示例覆盖全流程：创建集合 → 插入向量 → 相似度搜索 → Flush 持久化 → 跨 Segment 搜索 → FlightSQL SQL 查询 → 软删除 → Compact → 集合管理。

### Rust 客户端

```rust
use std::collections::HashMap;
use vdb_client::client::{MetadataColumnValue, VdbClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = VdbClient::connect("http://localhost:50051").await?;

    // 创建集合
    client
        .create_collection("docs", 768, "cosine", vec![("category", "string")])
        .await?;

    // 插入向量
    let ids = vec!["doc_1".into(), "doc_2".into()];
    let vectors = vec![vec![0.1; 768], vec![0.2; 768]];
    let mut metadata = HashMap::new();
    metadata.insert(
        "category".into(),
        vec![
            MetadataColumnValue::String("science".into()),
            MetadataColumnValue::String("tech".into()),
        ],
    );
    client.insert("docs", ids, vectors, metadata).await?;

    // 搜索 Top-10
    let hits = client
        .search("docs", vec![0.1; 768], 10, "cosine", "")
        .await?;
    for hit in &hits {
        println!("{}: {:.4}", hit.id, hit.distance);
    }

    Ok(())
}
```

### 嵌入式使用（无需服务器）

```rust
use std::collections::HashMap;
use vdb_common::config::*;
use vdb_common::metrics::DistanceMetric;
use vdb_storage::engine::StorageEngine;

let engine = StorageEngine::open("./data".as_ref())?;

engine.create_collection(CollectionConfig {
    name: "my_col".into(),
    dimension: 128,
    distance_metric: DistanceMetric::Cosine,
    index_config: Default::default(),
    metadata_fields: vec![],
    storage_format: Default::default(), // Vortex (default) or Parquet
})?;

engine.insert("my_col", ids, vectors, HashMap::new())?;
engine.flush("my_col")?;

let hits = engine.search("my_col", &query, 10, 0)?;
```

### grpcurl

```bash
# 创建集合
grpcurl -plaintext -d '{
  "name": "test", "dimension": 4, "distance_metric": "cosine"
}' localhost:50051 vdb.v1.VdbService/CreateCollection

# 插入
grpcurl -plaintext -d '{
  "collection": "test",
  "ids": ["a", "b"],
  "vectors": [{"values": [1,0,0,0]}, {"values": [0,1,0,0]}]
}' localhost:50051 vdb.v1.VdbService/Insert

# 搜索
grpcurl -plaintext -d '{
  "collection": "test",
  "query_vector": {"values": [1,0,0,0]},
  "top_k": 5
}' localhost:50051 vdb.v1.VdbService/Search

# 删除
grpcurl -plaintext -d '{
  "collection": "test",
  "ids": ["a"]
}' localhost:50051 vdb.v1.VdbService/Delete

# 手动 Flush
grpcurl -plaintext -d '{
  "collection": "test"
}' localhost:50051 vdb.v1.VdbService/Flush

# 手动 Compact
grpcurl -plaintext -d '{
  "collection": "test"
}' localhost:50051 vdb.v1.VdbService/Compact
```

## gRPC API

| RPC | 说明 |
|-----|------|
| `CreateCollection` | 创建集合，指定维度、距离度量、metadata schema |
| `ListCollections` | 列出所有集合 |
| `DropCollection` | 删除集合及其数据 |
| `Insert` | 批量插入向量 + 元数据 |
| `Search` | 向量相似度搜索 (Top-K) |
| `Delete` | 按 ID 软删除向量 |
| `Flush` | 手动触发 MemTable 刷盘 |
| `Compact` | 合并 Segment + 物理清理已删除数据 |

同时支持 **FlightSQL** 协议，可用标准 Arrow Flight 客户端执行 SQL：

```sql
SELECT _id, category, score FROM my_collection LIMIT 10
```

## 距离度量

| 度量 | 说明 | 别名 |
|------|------|------|
| `cosine` | 余弦距离 (默认) | — |
| `l2` | 欧几里得距离 | `euclidean` |
| `inner_product` | 负内积 | `ip`, `dot` |

## 项目结构

```
fdap-vdb/
├── proto/vdb/v1/service.proto    # gRPC 服务定义
├── crates/
│   ├── vdb-common/               # 共享类型、配置、错误、Schema
│   ├── vdb-index/                # 向量索引 (HNSW + brute-force)
│   ├── vdb-storage/              # 存储引擎 (WAL, MemTable, Vortex/Parquet Segment)
│   ├── vdb-query/                # DataFusion 集成 (TableProvider + UDF)
│   ├── vdb-server/               # gRPC + FlightSQL 服务
│   └── vdb-client/               # Rust 客户端 SDK
├── src/main.rs                   # CLI 入口
├── examples/demo.rs              # 完整使用示例
├── tests/                        # 集成测试
└── benches/                      # 性能基准
```

Crate 依赖链：`common` ← `index` ← `storage` ← `query` ← `server`；`client` 独立。

## 存储结构

```
data/
├── catalog.json                                  # 集合元数据
├── wal/{collection}/{collection}.wal             # 预写日志 (Arrow IPC)
└── collections/{collection}/
    ├── deleted_ids.json                          # 软删除 ID 列表
    └── segments/{seg_id}/
        ├── data.vortex                           # 向量 + 元数据 (Vortex, 默认)
        ├── data.parquet                          # 向量 + 元数据 (Parquet ZSTD, 可选)
        └── hnsw_index/                           # HNSW 索引文件
```

## 性能

测试环境：Apple Silicon, Rust release build, Criterion 基准测试。

### HNSW 索引搜索延迟 (Top-10)

| 向量数 | 维度 | 延迟 |
|--------|------|------|
| 10K | 128 | 168 us |
| 100K | 128 | 228 us |
| 10K | 768 | 1.19 ms |
| 100K | 768 | 1.53 ms |

### 存储引擎插入吞吐量 (WAL + MemTable)

| 批量大小 | 维度 | 吞吐量 |
|----------|------|--------|
| 1K | 128 | 172K vec/s |
| 10K | 128 | 791K vec/s |
| 1K | 768 | 95K vec/s |

### 存储引擎搜索延迟 (Top-10)

| 路径 | 向量数 | 维度 | 延迟 |
|------|--------|------|------|
| Segment (HNSW + Parquet) | 10K | 128 | 428 us |
| Segment (HNSW + Parquet) | 100K | 128 | 2.04 ms |
| Segment (HNSW + Parquet) | 10K | 768 | 1.44 ms |

```bash
cargo bench
```

## 配置

### HNSW 索引参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_nb_connection` | 16 | 每层最大连接数 (M) |
| `max_elements` | 1,000,000 | 最大向量数 |
| `ef_construction` | 200 | 构建时搜索范围 |
| `ef_search` | 50 | 查询时搜索范围 (越大越准但越慢) |

### 自动 Compact

服务器启动后会运行后台任务，定期检查每个集合并在满足条件时自动触发 Compact。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `check_interval_secs` | 60 | 检查间隔（秒）|
| `max_segments` | 4 | Segment 数量 ≥ 此值时触发 |
| `delete_ratio_threshold` | 0.2 | 已删除比例 > 此值时触发 |

触发条件（满足任一）：
- Segment 数量 ≥ `max_segments`
- 已删除行数 / 总行数 > `delete_ratio_threshold`

### Metadata 支持的字段类型

`string`, `int64`, `float64`, `bool`

## 开发

```bash
# 构建
cargo build --workspace

# 测试
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings

# 格式化
cargo fmt --all -- --check

# 基准测试
cargo bench

# 运行示例
cargo run --example demo
```

## 技术栈

| 组件 | 版本 | 角色 |
|------|------|------|
| Arrow | 57 | 内存列式格式，向量 `FixedSizeList<Float32>` |
| Vortex | 0.60 | 默认持久化格式，高性能列式存储 |
| Parquet | 57 | 可选持久化格式，ZSTD 压缩，RowSelection |
| DataFusion | 52 | SQL 查询引擎，TableProvider + UDF |
| Arrow Flight | 57 | 网络传输，FlightSQL 协议 |
| tonic | 0.14 | gRPC 框架 |
| hnsw_rs | 0.3 | HNSW 向量索引 |

## License

MIT
