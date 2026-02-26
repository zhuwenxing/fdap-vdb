# fdap-vdb 技术原理

本文档详细介绍 fdap-vdb 的内部架构和各组件的工作原理。

## 目录

- [FDAP 架构概述](#fdap-架构概述)
- [数据模型](#数据模型)
- [写入路径](#写入路径)
- [搜索路径](#搜索路径)
- [存储引擎](#存储引擎)
- [向量索引](#向量索引)
- [查询引擎](#查询引擎)
- [网络服务](#网络服务)
- [关键设计决策](#关键设计决策)

---

## FDAP 架构概述

FDAP 是 **F**light + **D**ataFusion + **A**rrow + **P**arquet 的缩写，由 InfluxDB 团队推广的现代数据系统架构。fdap-vdb 将这套架构应用于向量数据库场景：

```
                         ┌─────────────────────────────────┐
                         │          Client                  │
                         │   (gRPC SDK / FlightSQL Client)  │
                         └──────────┬──────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
           ┌──────────────┐               ┌──────────────────┐
           │  gRPC 服务    │               │  FlightSQL 服务   │
           │  (tonic)      │               │  (arrow-flight)   │
           │              │               │                   │
           │ Insert/Search │               │  SQL 查询          │
           │ CRUD 操作     │               │  Arrow 零拷贝传输   │
           └──────┬───────┘               └────────┬──────────┘
                  │                                │
                  │          ┌──────────────┐      │
                  └─────────►│ DataFusion   │◄─────┘
                             │ 查询引擎      │
                             │              │
                             │ TableProvider │
                             │ 距离 UDF     │
                             └──────┬───────┘
                                    │
                          ┌─────────▼──────────┐
                          │   Storage Engine    │
                          │                     │
                          │  ┌──────┐ ┌───────┐ │
                          │  │ WAL  │ │MemTable│ │
                          │  └──┬───┘ └───┬───┘ │
                          │     │         │     │
                          │  ┌──▼─────────▼──┐  │
                          │  │   Segment     │  │
                          │  │               │  │
                          │  │ ┌──────────┐  │  │
                          │  │ │ Parquet  │  │  │
                          │  │ │ (Arrow)  │  │  │
                          │  │ └──────────┘  │  │
                          │  │ ┌──────────┐  │  │
                          │  │ │ HNSW     │  │  │
                          │  │ │ Index    │  │  │
                          │  │ └──────────┘  │  │
                          │  └───────────────┘  │
                          └─────────────────────┘
```

各组件的职责：

| 组件 | FDAP 角色 | 在 fdap-vdb 中的体现 |
|------|-----------|---------------------|
| **Arrow** | 内存数据格式 | 向量用 `FixedSizeList<Float32>` 表示，所有数据以 `RecordBatch` 为单位在各模块间流转，零序列化开销 |
| **Parquet** | 持久化存储 | 每个 Segment 的向量+元数据存为一个 Parquet 文件，ZSTD 压缩，搜索时利用 RowSelection + 列投影避免全表扫描 |
| **DataFusion** | 查询引擎 | 每个集合注册为 `TableProvider`，支持 SQL 查询；自定义 `cosine_distance` / `l2_distance` / `inner_product_distance` UDF |
| **Flight** | 网络传输 | FlightSQL 协议支持 SQL 查询结果以 Arrow RecordBatch 流式传输，客户端零反序列化接收 |

---

## 数据模型

### Arrow Schema

每个集合有一个固定的 Arrow Schema，包含系统列和用户自定义元数据列：

```
┌─────────────┬─────────────────────────────┬──────────┐
│ 列名         │ Arrow 类型                   │ nullable │
├─────────────┼─────────────────────────────┼──────────┤
│ _id         │ Utf8                        │ false    │  ← 向量标识符
│ vector      │ FixedSizeList<Float32>(dim) │ false    │  ← 向量数据
│ {metadata}  │ Utf8 / Int64 / Float64 / Boolean │ true │  ← 用户自定义
│ _created_at │ Timestamp(Microsecond)      │ false    │  ← 插入时间
│ _deleted    │ Boolean                     │ false    │  ← 软删除标记
└─────────────┴─────────────────────────────┴──────────┘
```

**为什么用 `FixedSizeList<Float32>`**：这是 Arrow 生态存储 embedding 的标准做法（LanceDB、ChromaDB 均采用），维度信息编码在类型定义中，单个向量的内存布局是连续的 `dim * 4` 字节，可直接传给 SIMD 距离计算函数，无需额外转换。

### 距离度量

```
Cosine Distance  = 1 - (a · b) / (|a| * |b|)    值域 [0, 2]，0 = 完全相同
L2 Distance      = sqrt(Σ(aᵢ - bᵢ)²)            值域 [0, +∞)
Inner Product    = -(a · b)                       取负使"更相似 = 更小"
```

Cosine 是默认度量，也是 RAG 场景最常用的。在 HNSW 索引内部，Cosine 通过"归一化 + Dot Product"等价实现，避免每次搜索时重复计算范数。

---

## 写入路径

```
Client Insert(ids, vectors, metadata)
    │
    ▼
┌──────────────────────────────┐
│ 1. 维度校验                    │  向量长度 != collection.dimension → 报错
│    ids.len() == vectors.len() │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 2. 构造 RecordBatch            │  ids → StringArray
│    (build_record_batch)       │  vectors → FixedSizeListArray<Float32>
│                               │  metadata → 各类型列
│                               │  _created_at → 当前微秒时间戳
│                               │  _deleted → 全 false
└──────────┬───────────────────┘
           │
           ├───► WAL.append(batch)     ← 3. 持久化到预写日志 (fsync)
           │
           └───► MemTable.insert(batch) ← 4. 追加到内存缓冲区
```

### WAL（预写日志）

WAL 采用 **Arrow IPC Stream** 格式，每条记录的物理布局：

```
┌──────────┬─────────────────────────┐
│ len: u64 │ Arrow IPC Stream 帧      │
│ (8 bytes)│ (schema + batch + EOS)  │
├──────────┼─────────────────────────┤
│ len: u64 │ Arrow IPC Stream 帧      │
│          │                         │
└──────────┴─────────────────────────┘
```

每次 `append` 调用 `fsync` 确保落盘。选择 Arrow IPC 而非 protobuf/bincode 是因为 RecordBatch 可以零拷贝序列化，且恢复时不需要 schema 转换。

### MemTable（内存表）

MemTable 是一个 `RwLock<Vec<RecordBatch>>` — 每次 insert 直接追加，不做排序或合并。搜索时暴力扫描所有 batch。当行数超过阈值（默认 100K）时触发 flush。

### Flush 流程

```
MemTable.freeze()        ← 取出所有 batch，重置为空
    │
    ▼
concat_batches()         ← 合并为单个 RecordBatch
    │
    ├───► ArrowWriter → data.parquet     ← ZSTD 压缩写入 Parquet
    │
    ├───► build_hnsw_index → hnsw_index/ ← 对 vector 列构建 HNSW 索引并序列化
    │
    ├───► Catalog.register_segment()     ← 注册新 Segment 到元数据
    │
    └───► WAL.clear()                    ← 清除已持久化的 WAL
```

Flush 产出的 Segment 是不可变的（immutable），一旦写入就不再修改。这简化了并发控制 — 读取不需要加锁。

---

## 搜索路径

```
Client Search(query_vector, top_k)
    │
    ▼
┌─────────────────────────────────────────────────┐
│               对每个 Segment 并行                  │
│                                                  │
│  HNSW Index ──search(query, top_k, ef)──► [(row_id, distance)]
│       │                                          │
│       │  row_ids 按升序排序                         │
│       ▼                                          │
│  Parquet Reader                                  │
│       │  ProjectionMask: 跳过 vector 列            │
│       │  RowSelection: 只读 HNSW 返回的行           │
│       ▼                                          │
│  RecordBatch (仅 _id + metadata + _deleted)       │
│       │                                          │
│       │  过滤 _deleted = true                      │
│       │  提取 id + metadata                       │
│       ▼                                          │
│  Vec<SearchHit>                                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│               MemTable 暴力搜索                    │
│                                                  │
│  对每个 batch:                                     │
│    FixedSizeListArray → 逐行计算 distance          │
│    过滤 _deleted                                  │
│    收集 (id, distance, metadata)                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
           合并所有 hits
           按 distance 升序排序
           取前 top_k 个
                     │
                     ▼
           Vec<SearchHit> 返回客户端
```

### Parquet 读取优化

搜索时从 Parquet 读取数据的关键优化（使搜索延迟从 141ms 降到 2ms）：

**RowSelection** — HNSW 返回 10 个 row_id（如 [3, 47, 102, ...]）后，构造一个 RowSelection 告诉 Parquet reader 跳到这些行直接读取：

```
row_ids = [3, 47, 102]

RowSelector::skip(3)      ← 跳过行 0-2
RowSelector::select(1)    ← 读取行 3
RowSelector::skip(43)     ← 跳过行 4-46
RowSelector::select(1)    ← 读取行 47
RowSelector::skip(54)     ← 跳过行 48-101
RowSelector::select(1)    ← 读取行 102
```

**列投影** — 搜索结果不需要 vector 列（距离已由 HNSW 返回），通过 `ProjectionMask::roots` 排除 vector 列，省去读取最大列的 I/O。对于 768 维向量，vector 列占每行 3072 字节（768 * 4），而 _id + metadata 通常只有几十字节。

---

## 存储引擎

### 整体结构

```rust
StorageEngine
├── data_dir: PathBuf
├── catalog: RwLock<Catalog>         ← 全局元数据 (catalog.json)
└── collections: RwLock<HashMap<String, Arc<RwLock<CollectionState>>>>
                                      │
                                      └── CollectionState
                                          ├── config: CollectionConfig
                                          ├── schema: SchemaRef
                                          ├── wal: Wal
                                          ├── memtable: MemTable
                                          └── segments: Vec<Segment>
                                                          │
                                                          └── Segment
                                                              ├── data.parquet
                                                              └── index: Arc<HnswIndex>
```

### Catalog（元数据目录）

`catalog.json` 持久化所有集合的配置和 Segment 列表：

```json
{
  "collections": {
    "articles": {
      "config": {
        "name": "articles",
        "dimension": 768,
        "distance_metric": "cosine",
        "metadata_fields": [{"name": "category", "field_type": "string"}]
      },
      "segments": [
        {"id": "80a866c1-...", "num_rows": 10000, "has_index": true}
      ],
      "next_row_id": 10000
    }
  }
}
```

每次创建/删除集合、注册新 Segment 时都会同步写入 `catalog.json`。这是当前最简单的实现，未来可升级为 WAL 或嵌入式 KV 存储。

### 磁盘布局

```
data/
├── catalog.json
├── wal/
│   └── articles/
│       └── articles.wal                 ← Arrow IPC 帧序列
└── collections/
    └── articles/
        └── segments/
            └── 80a866c1-.../
                ├── data.parquet         ← 向量 + 元数据 (ZSTD 压缩)
                └── hnsw_index/
                    ├── hnsw.hnsw.data   ← 向量数据 (hnsw_rs 格式)
                    └── hnsw.hnsw.graph  ← 图结构 (hnsw_rs 格式)
```

### 崩溃恢复

重启时的恢复流程：

1. 读取 `catalog.json` → 获取所有集合配置和 Segment 列表
2. 对每个集合：
   - 加载所有 Segment（读 Parquet metadata + mmap HNSW 索引）
   - 读取 WAL → 回放到新的 MemTable
3. 服务就绪

WAL 中的数据是 flush 之前未持久化的写入。如果 flush 成功后 WAL 已清除，则恢复时 WAL 为空。如果 flush 中途崩溃，WAL 仍保留，数据不丢失。

---

## 向量索引

### HNSW 算法

HNSW（Hierarchical Navigable Small World）是当前工业界最广泛使用的近似最近邻搜索算法，由 Malkov & Yashunin 于 2016 年提出。

核心思想是构建一个多层图结构：

```
Layer 2:    o ──────────── o              (稀疏，长距离连接)
            |              |
Layer 1:    o ─── o ────── o ─── o        (中等密度)
            |     |        |     |
Layer 0:    o─o─o─o─o─o─o─o─o─o─o─o      (最稠密，所有节点)
```

- **插入**：随机选择一个最高层级 L，从最顶层贪心搜索到 L，然后在 L 到 0 层逐层插入并建立双向连接
- **搜索**：从最顶层的入口点开始，逐层贪心搜索，到达 Layer 0 后执行 beam search 返回 top-k

关键参数：

| 参数 | 含义 | 默认值 | 影响 |
|------|------|--------|------|
| M (`max_nb_connection`) | 每层最大连接数 | 16 | 越大越准但内存越多 |
| ef_construction | 建索引时的搜索范围 | 200 | 越大索引质量越高但构建越慢 |
| ef_search | 查询时的搜索范围 | 50 | 越大越准但查询越慢 |

### hnsw_rs 封装

fdap-vdb 使用 `hnsw_rs` crate 实现 HNSW，但做了以下适配：

**距离度量映射** — `hnsw_rs` 没有原生的 Cosine distance，通过等价变换实现：

```
对归一化向量: cosine_distance(a, b) = 1 - dot(a, b)
```

所以 Cosine 和 InnerProduct 都映射到 `DistDot`，Cosine 在插入和搜索前先对向量做 L2 归一化。

```rust
enum HnswInner {
    L2(Hnsw<'static, f32, DistL2>),     // L2 度量
    Dot(Hnsw<'static, f32, DistDot>),    // Cosine 和 InnerProduct
}
```

**生命周期处理** — `hnsw_rs` 从文件加载索引时，`HnswIo` 通过 mmap 映射文件，`Hnsw` 引用 `HnswIo` 的数据。这导致 `Hnsw<'b>` 的生命周期与 `HnswIo` 绑定。为了将两者存入同一个结构体，使用 `Box::into_raw` 将 `HnswIo` 的生命周期提升为 `'static`：

```rust
let io_box = Box::new(HnswIo::new(dir, "hnsw"));
let io_ptr = Box::into_raw(io_box);
let io_ref: &'static mut HnswIo = unsafe { &mut *io_ptr };
let hnsw: Hnsw<'static, f32, DistL2> = io_ref.load_hnsw()?;
let io_box = unsafe { Box::from_raw(io_ptr) };  // 重新持有所有权
```

**持久化** — `hnsw_rs` 的 `file_dump(dir, basename)` 将图结构和向量数据分别写入两个文件。加载时通过 `HnswIo::new(dir, basename).load_hnsw()` 恢复。

---

## 查询引擎

### DataFusion 集成

每个集合注册为 DataFusion 的 `TableProvider`，使得集合可以直接用 SQL 查询：

```sql
SELECT _id, category, cosine_distance(vector, ?) AS dist
FROM articles
WHERE category = 'science'
ORDER BY dist ASC
LIMIT 10;
```

**`CollectionTableProvider`** 实现 `TableProvider` trait，`scan()` 方法将集合的所有数据（Segment + MemTable）加载为内存中的 `MemTable` 交给 DataFusion 执行。

### 向量距离 UDF

注册了三个标量 UDF，均实现 `ScalarUDFImpl` trait：

| UDF 名称 | 距离度量 | 签名 |
|----------|---------|------|
| `cosine_distance(vector, query)` | Cosine | `(FixedSizeList, FixedSizeList) → Float32` |
| `l2_distance(vector, query)` | L2 | `(FixedSizeList, FixedSizeList) → Float32` |
| `inner_product_distance(vector, query)` | Inner Product | `(FixedSizeList, FixedSizeList) → Float32` |

UDF 内部工作原理：
1. 从两个 `FixedSizeListArray` 中提取底层 `Float32Array`
2. 按 `dim` 步长遍历每行
3. 对每对 `(v[i*dim..(i+1)*dim], q[i*dim..(i+1)*dim])` 调用 `DistanceMetric::compute`
4. 返回 `Float32Array`

---

## 网络服务

服务器在同一端口上同时提供两个协议：

### gRPC 服务 (VdbService)

通过 `proto/vdb/v1/service.proto` 定义，使用 `tonic-prost-build` 编译生成 Rust 代码。提供 6 个 RPC：

```protobuf
service VdbService {
  rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc ListCollections(ListCollectionsRequest)   returns (ListCollectionsResponse);
  rpc DropCollection(DropCollectionRequest)     returns (DropCollectionResponse);
  rpc Insert(InsertRequest)                     returns (InsertResponse);
  rpc Search(SearchRequest)                     returns (SearchResponse);
  rpc Delete(DeleteRequest)                     returns (DeleteResponse);
}
```

Insert 的数据流：`protobuf Vector → Vec<Vec<f32>> → Arrow RecordBatch → WAL + MemTable`

Search 的数据流：`protobuf SearchRequest → engine.search() → Vec<SearchHit> → protobuf SearchResponse`

### FlightSQL 服务

实现 `arrow_flight::sql::server::FlightSqlService` trait，支持标准 FlightSQL 协议。

SQL 查询的完整流程：

```
Client                              Server
  │                                    │
  │── execute(SQL) ───────────────────►│ get_flight_info_statement()
  │                                    │   1. ctx.sql(query) → 验证 SQL 有效
  │                                    │   2. 提取结果 schema
  │                                    │   3. TicketStatementQuery { handle: SQL }
  │                                    │   4. ticket.as_any().encode_to_vec()
  │◄── FlightInfo { schema, ticket } ──│      ↑ 必须包装为 protobuf Any
  │                                    │
  │── do_get(ticket) ─────────────────►│ do_get_statement()
  │                                    │   1. 解码 ticket → 取出 SQL
  │                                    │   2. ctx.sql(query).collect()
  │                                    │   3. FlightDataEncoderBuilder 编码
  │◄── Stream<FlightData> ────────────│      Arrow RecordBatch 零拷贝传输
```

ticket 编码的关键：必须通过 `ProstMessageExt::as_any()` 将 `TicketStatementQuery` 包装为 protobuf `Any` 消息，FlightSQL server 框架依赖 `type_url` 将 `do_get` 请求路由到正确的 handler。

### tonic 0.14 生态

arrow-flight 57 依赖 tonic 0.14，该版本将 prost 集成拆分为独立 crate：

```
tonic 0.14          ← gRPC 框架核心
tonic-prost 0.14    ← 运行时编解码 (ProstCodec)
tonic-build 0.14    ← proto 编译器驱动
tonic-prost-build 0.14 ← proto → Rust 代码生成
```

`build.rs` 中使用 `tonic_prost_build::configure()` 而非已移除的 `tonic_build::configure()`。

---

## 关键设计决策

### 追加式不可变 Segment

每次 flush 产生一个新的不可变 Segment，不修改已有 Segment。

**优点**：
- 契合 Parquet 的不可变特性（Parquet 文件一旦写入就不应修改）
- 简化并发控制（读不需要锁）
- HNSW 索引与 Parquet 数据保持一致，无需同步更新

**代价**：
- 删除采用软删除（`_deleted` 标记），空间不立即回收
- 随着 Segment 增多，搜索需扫描更多索引（未来需要 compaction）

### Arrow 作为内部数据总线

所有模块间的数据传输统一使用 Arrow `RecordBatch`：

```
Client protobuf ──► RecordBatch ──► WAL (Arrow IPC)
                        │
                        ├──► MemTable (Vec<RecordBatch>)
                        │
                        ├──► Parquet (ArrowWriter)
                        │
                        ├──► HNSW (从 FixedSizeListArray 提取 f32 切片)
                        │
                        └──► DataFusion (MemTable / TableProvider)
```

无需在不同格式间反复转换，减少了序列化开销和内存拷贝。

### WAL 使用 Arrow IPC

WAL 格式选择 Arrow IPC Stream 而非 protobuf 或自定义二进制格式：

- 与内存中的 RecordBatch 完全同构，恢复时零转换
- 自带 schema 信息，每条记录自描述
- 可以用标准 Arrow IPC reader 工具直接读取 WAL 内容（便于调试）

### HNSW Cosine = 归一化 + Dot Product

`hnsw_rs` 不直接支持 Cosine distance，但提供了 `DistDot`。数学上：

```
cosine_similarity(a, b)  = dot(a, b) / (|a| * |b|)
                         = dot(normalize(a), normalize(b))

cosine_distance(a, b)    = 1 - cosine_similarity(a, b)
```

对归一化向量，`DistDot` 返回 `1 - dot(a, b)` = cosine_distance。所以 Cosine 模式下：
- insert 时先归一化再插入
- search 时先归一化 query 再搜索
- `DistDot` 的返回值直接就是 cosine distance

### Parquet RowSelection 优化搜索

搜索路径的瓶颈是从 Parquet 读取 HNSW 命中行的元数据。朴素实现（全表扫描 + take）在 100K 行 128 维时耗时 141ms。

优化后使用 Parquet 的 `RowSelection` + `ProjectionMask`：
- 只读 HNSW 返回的 ~10 行（跳过其余 99,990 行）
- 跳过 vector 列（占总数据量 90%+）
- 100K 行 128 维：141ms → 2.04ms（-98.6%）

这也是 Parquet 格式的核心优势之一 — 行组级别的元数据和偏移索引使得精准读取少量行成为可能，不像行式存储需要扫描定位。
