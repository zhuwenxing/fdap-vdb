# fdap-vdb

基于 FDAP（Flight + DataFusion + Arrow + Parquet）技术栈的向量数据库。

## 技术栈

- **语言**: Rust (edition 2021)
- **Arrow/Parquet**: v57 (由 DataFusion 52 约束)
- **DataFusion**: v52
- **gRPC**: tonic 0.14 / arrow-flight 57
- **向量索引**: hnsw_rs 0.3

## 项目结构

Cargo workspace，crates 依赖链：`common` ← `index` ← `storage` ← `query` ← `server`；`client` 独立。

## 常用命令

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
cargo run -- serve --data-dir ./data --port 50051
cargo run --example demo          # 完整使用示例
cargo bench                        # 性能基准测试
```

## 约定

- 向量存储类型: `FixedSizeList<Float32>`
- Schema 保留列: `_id` (Utf8), `vector` (FixedSizeList), `_created_at` (TimestampMicrosecond), `_deleted` (Boolean)
- 距离度量默认 Cosine
- 提交使用 `git commit -sm` 带签名
