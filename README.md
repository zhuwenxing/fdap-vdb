# fdap-vdb

基于 **FDAP**（Flight + DataFusion + Arrow + Parquet/Vortex）技术栈构建的嵌入式向量数据库，面向 RAG / 语义搜索场景。

## 安装

```bash
# 从 GitHub Release 下载预编译 wheel（无需 Rust toolchain）
uv pip install "https://github.com/zhuwenxing/fdap-vdb/releases/latest/download/fdap_vdb-0.1.0-cp312-cp312-macosx_11_0_arm64.whl"

# 或从源码编译安装（需要 Rust toolchain）
uv pip install "fdap-vdb @ git+https://github.com/zhuwenxing/fdap-vdb.git#subdirectory=crates/vdb-python"
```

## 快速开始

```python
import numpy as np
from fdap_vdb import VDB

# 打开数据库（目录不存在会自动创建）
db = VDB("./my_data")

# 创建集合：指定维度、距离度量、metadata 字段
db.create_collection(
    "docs",
    dimension=768,
    distance_metric="cosine",                        # cosine (默认) | l2 | inner_product
    metadata_fields=[("category", "string"), ("score", "int64")],
)

# 插入向量（支持 numpy 数组或 list）
vectors = np.random.rand(100, 768).astype(np.float32)
ids = [f"doc_{i}" for i in range(100)]
db.insert(
    "docs",
    ids=ids,
    vectors=vectors,
    metadata={
        "category": ["tech"] * 50 + ["science"] * 50,
        "score": list(range(100)),
    },
)

# 搜索 Top-10
results = db.search("docs", query_vector=np.random.rand(768).astype(np.float32), top_k=10)
for r in results:
    print(f'{r["id"]}: distance={r["distance"]:.4f}, metadata={r["metadata"]}')

# SQL 查询（返回 pyarrow.RecordBatch）
batch = db.sql("SELECT _id, category, score FROM docs WHERE score > 50 ORDER BY score DESC")
print(batch.to_pandas())

# 删除
db.delete("docs", ids=["doc_0", "doc_1"])

# 持久化 & 合并
db.flush("docs")
db.compact("docs")

# 集合管理
db.list_collections()   # [{"name": "docs", "dimension": 768, ...}]
db.drop_collection("docs")
```

## API

### `VDB(data_dir: str)`

打开或创建数据库。

### `create_collection(name, dimension, distance_metric=None, metadata_fields=None)`

创建集合。`distance_metric` 可选 `"cosine"` (默认)、`"l2"`、`"inner_product"`。
`metadata_fields` 为 `[(name, type)]` 列表，type 支持 `string`、`int64`、`float64`、`bool`。

### `insert(collection, ids, vectors, metadata=None) -> int`

批量插入。`vectors` 接受 numpy 2D 数组或 `list[list[float]]`。返回插入数量。

### `search(collection, query_vector, top_k=10) -> list[dict]`

向量相似度搜索。`query_vector` 接受 numpy 1D 数组或 `list[float]`。
返回 `[{"id", "distance", "metadata"}]`。

### `delete(collection, ids) -> int`

按 ID 软删除。返回实际删除数量。

### `flush(collection)`

手动将 MemTable 刷盘为 Segment。

### `compact(collection) -> dict`

合并 Segment + 物理清理已删除行。返回 `{"segments_before", "segments_after", "rows_removed"}`。

### `sql(query) -> pyarrow.RecordBatch`

执行 SQL 查询，每个集合自动注册为表。

### `list_collections() -> list[dict]`

列出所有集合。

### `drop_collection(name)`

删除集合及其所有数据。

## 特性

- **HNSW 向量索引** — 支持 Cosine / L2 / Inner Product
- **双存储格式** — Vortex (默认) 和 Parquet，可按集合切换
- **WAL** — Arrow IPC 预写日志，crash recovery
- **SQL 查询** — DataFusion 引擎，支持向量距离 UDF
- **软删除 + 自动 Compact** — 后台自动合并 Segment
- **gRPC + FlightSQL** — 支持 C/S 模式部署

## 开发

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo bench
```

## License

MIT
