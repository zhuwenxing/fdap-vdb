use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use vdb_common::config::{CollectionConfig, IndexConfig, MetadataFieldConfig, MetadataFieldType};
use vdb_common::metrics::DistanceMetric;
use vdb_storage::engine::{MetadataColumnValue, StorageEngine};

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn make_config(name: &str, dim: usize) -> CollectionConfig {
    CollectionConfig {
        name: name.to_string(),
        dimension: dim,
        distance_metric: DistanceMetric::Cosine,
        index_config: IndexConfig {
            max_nb_connection: 16,
            max_elements: 200_000,
            ef_construction: 200,
            ef_search: 50,
        },
        metadata_fields: vec![
            MetadataFieldConfig {
                name: "category".to_string(),
                field_type: MetadataFieldType::String,
            },
            MetadataFieldConfig {
                name: "score".to_string(),
                field_type: MetadataFieldType::Int64,
            },
        ],
    }
}

fn make_metadata(n: usize) -> HashMap<String, Vec<MetadataColumnValue>> {
    let mut rng = rand::thread_rng();
    let cats = ["science", "tech", "art", "music", "sports"];
    let mut meta = HashMap::new();
    meta.insert(
        "category".to_string(),
        (0..n)
            .map(|_| MetadataColumnValue::String(cats[rng.gen_range(0..cats.len())].to_string()))
            .collect(),
    );
    meta.insert(
        "score".to_string(),
        (0..n)
            .map(|_| MetadataColumnValue::Int64(rng.gen_range(0..100)))
            .collect(),
    );
    meta
}

/// Benchmark StorageEngine insert throughput (WAL + MemTable).
fn bench_engine_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_insert");
    group.sample_size(10);

    for &(n, dim) in &[(1_000, 128), (10_000, 128), (1_000, 768)] {
        let vecs = random_vectors(n, dim);
        let ids: Vec<String> = (0..n).map(|i| format!("id_{i}")).collect();
        let metadata = make_metadata(n);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("dim{dim}"), n),
            &(ids, vecs, metadata),
            |b, (ids, vecs, meta)| {
                b.iter_with_setup(
                    || {
                        let dir = tempfile::tempdir().unwrap();
                        let engine = StorageEngine::open(dir.path()).unwrap();
                        engine
                            .create_collection(make_config("bench", dim))
                            .unwrap();
                        (dir, engine)
                    },
                    |(_dir, engine)| {
                        engine
                            .insert("bench", ids.clone(), vecs.clone(), meta.clone())
                            .unwrap();
                    },
                );
            },
        );
    }
    group.finish();
}

/// Benchmark StorageEngine flush (MemTable → Parquet + HNSW build).
fn bench_engine_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_flush");
    group.sample_size(10);

    for &(n, dim) in &[(1_000, 128), (10_000, 128), (1_000, 768)] {
        group.bench_with_input(
            BenchmarkId::new(format!("dim{dim}"), n),
            &(n, dim),
            |b, &(n, dim)| {
                b.iter_with_setup(
                    || {
                        let dir = tempfile::tempdir().unwrap();
                        let engine = StorageEngine::open(dir.path()).unwrap();
                        engine
                            .create_collection(make_config("bench", dim))
                            .unwrap();
                        let vecs = random_vectors(n, dim);
                        let ids: Vec<String> = (0..n).map(|i| format!("id_{i}")).collect();
                        let metadata = make_metadata(n);
                        engine
                            .insert("bench", ids, vecs, metadata)
                            .unwrap();
                        (dir, engine)
                    },
                    |(_dir, engine)| {
                        engine.flush("bench").unwrap();
                    },
                );
            },
        );
    }
    group.finish();
}

/// Benchmark StorageEngine search — memtable (brute-force) vs segment (HNSW).
fn bench_engine_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_search");

    for &(n, dim) in &[(10_000, 128), (100_000, 128), (10_000, 768)] {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine
            .create_collection(make_config("bench", dim))
            .unwrap();

        let vecs = random_vectors(n, dim);
        let ids: Vec<String> = (0..n).map(|i| format!("id_{i}")).collect();
        let metadata = make_metadata(n);
        engine
            .insert("bench", ids, vecs, metadata)
            .unwrap();

        let mut rng = rand::thread_rng();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Search memtable (brute-force)
        group.bench_with_input(
            BenchmarkId::new(format!("memtable/dim{dim}/top10"), n),
            &query,
            |b, q| {
                b.iter(|| {
                    engine.search("bench", q, 10, 0).unwrap();
                });
            },
        );

        // Flush and search segment (HNSW)
        engine.flush("bench").unwrap();

        group.bench_with_input(
            BenchmarkId::new(format!("segment/dim{dim}/top10"), n),
            &query,
            |b, q| {
                b.iter(|| {
                    engine.search("bench", q, 10, 0).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_engine_insert,
    bench_engine_flush,
    bench_engine_search
);
criterion_main!(benches);
