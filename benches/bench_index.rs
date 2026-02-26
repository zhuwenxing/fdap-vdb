use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use vdb_common::config::IndexConfig;
use vdb_common::metrics::DistanceMetric;
use vdb_index::hnsw::HnswIndex;
use vdb_index::traits::VectorIndex;

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn index_config(max_elements: usize) -> IndexConfig {
    IndexConfig {
        max_nb_connection: 16,
        max_elements,
        ef_construction: 200,
        ef_search: 50,
    }
}

/// Benchmark HNSW insert throughput at different scales and dimensions.
fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.sample_size(10);

    for &(n, dim) in &[(1_000, 128), (10_000, 128), (1_000, 768), (10_000, 768)] {
        let vecs = random_vectors(n, dim);
        let ids: Vec<u64> = (0..n as u64).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("dim{dim}"), n),
            &(ids, vec_refs),
            |b, (ids, vecs)| {
                b.iter(|| {
                    let idx = HnswIndex::new(dim, DistanceMetric::Cosine, &index_config(n + 1000));
                    idx.insert(ids, vecs).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark HNSW search latency at different scales and dimensions.
fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for &(n, dim) in &[(10_000, 128), (100_000, 128), (10_000, 768), (100_000, 768)] {
        // Build index once
        let vecs = random_vectors(n, dim);
        let ids: Vec<u64> = (0..n as u64).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let idx = HnswIndex::new(dim, DistanceMetric::Cosine, &index_config(n + 1000));
        idx.insert(&ids, &vec_refs).unwrap();

        // Query vector
        let query = random_vectors(1, dim).into_iter().next().unwrap();

        group.bench_with_input(
            BenchmarkId::new(format!("dim{dim}/top10"), n),
            &query,
            |b, q| {
                b.iter(|| {
                    idx.search(q, 10, 0).unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark the impact of different ef_search values on search latency.
fn bench_hnsw_ef_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_ef_search");

    let n = 100_000;
    let dim = 128;
    let vecs = random_vectors(n, dim);
    let ids: Vec<u64> = (0..n as u64).collect();
    let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let idx = HnswIndex::new(dim, DistanceMetric::Cosine, &index_config(n + 1000));
    idx.insert(&ids, &vec_refs).unwrap();

    let query = random_vectors(1, dim).into_iter().next().unwrap();

    for &ef in &[16, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("100k_dim128", ef), &ef, |b, &ef| {
            b.iter(|| {
                idx.search(&query, 10, ef).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_hnsw_ef_search
);
criterion_main!(benches);
