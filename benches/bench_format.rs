use std::sync::Arc;
use std::time::SystemTime;

use arrow::array::{
    BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use vdb_common::config::StorageFormat::*;
use vdb_storage::format;

/// Generate a RecordBatch with n rows and dim-dimensional vectors.
fn make_batch(n: usize, dim: usize) -> (Arc<Schema>, RecordBatch) {
    let mut rng = rand::thread_rng();

    let ids: Vec<String> = (0..n).map(|i| format!("id_{i}")).collect();
    let id_array = Arc::new(StringArray::from(ids)) as Arc<dyn arrow::array::Array>;

    let flat: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let values = Float32Array::from(flat);
    let vector_array = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(values),
            None,
        )
        .unwrap(),
    ) as Arc<dyn arrow::array::Array>;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let ts_array =
        Arc::new(TimestampMicrosecondArray::from(vec![now; n])) as Arc<dyn arrow::array::Array>;
    let deleted_array =
        Arc::new(BooleanArray::from(vec![false; n])) as Arc<dyn arrow::array::Array>;

    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new(
            "_created_at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("_deleted", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![id_array, vector_array, ts_array, deleted_array],
    )
    .unwrap();

    (schema, batch)
}

/// Benchmark: write RecordBatch to file (Parquet vs Vortex).
fn bench_format_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_write");
    group.sample_size(10);

    for &(n, dim) in &[(10_000, 128), (100_000, 128), (10_000, 768), (100_000, 768)] {
        let (schema, batch) = make_batch(n, dim);

        for &fmt in &[Parquet, Vortex] {
            let label = format!("{fmt}/{n}x{dim}d");
            group.bench_function(BenchmarkId::new(&label, n), |b| {
                b.iter_with_setup(
                    || {
                        let dir = tempfile::tempdir().unwrap();
                        let filename = match fmt {
                            Parquet => "data.parquet",
                            Vortex => "data.vortex",
                        };
                        let path = dir.path().join(filename);
                        (dir, path)
                    },
                    |(_dir, path)| match fmt {
                        Parquet => {
                            format::parquet::write_data(&path, schema.clone(), &batch).unwrap();
                        }
                        Vortex => {
                            format::vortex::write_data(&path, schema.clone(), &batch).unwrap();
                        }
                    },
                );
            });
        }
    }
    group.finish();
}

/// Benchmark: read all data from file (Parquet vs Vortex).
fn bench_format_read_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_read_all");
    group.sample_size(10);

    for &(n, dim) in &[(10_000, 128), (100_000, 128), (10_000, 768), (100_000, 768)] {
        let (schema, batch) = make_batch(n, dim);

        for &fmt in &[Parquet, Vortex] {
            let dir = tempfile::tempdir().unwrap();
            let filename = match fmt {
                Parquet => "data.parquet",
                Vortex => "data.vortex",
            };
            let path = dir.path().join(filename);

            // Write test data
            match fmt {
                Parquet => format::parquet::write_data(&path, schema.clone(), &batch).unwrap(),
                Vortex => format::vortex::write_data(&path, schema.clone(), &batch).unwrap(),
            }

            // Print file size for comparison
            let file_size = std::fs::metadata(&path).unwrap().len();
            println!(
                "[{fmt}] {n}x{dim}d file size: {:.2} MB",
                file_size as f64 / (1024.0 * 1024.0)
            );

            let label = format!("{fmt}/{n}x{dim}d");
            group.bench_function(BenchmarkId::new(&label, n), |b| {
                b.iter(|| {
                    match fmt {
                        Parquet => format::parquet::read_all(&path, &schema).unwrap(),
                        Vortex => format::vortex::read_all(&path, &schema).unwrap(),
                    };
                });
            });
        }
    }
    group.finish();
}

/// Benchmark: read metadata (non-vector columns) for specific row IDs (hot path).
fn bench_format_read_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_read_metadata");
    group.sample_size(10);

    let n = 100_000;
    for &dim in &[128, 768] {
        let (schema, batch) = make_batch(n, dim);

        // Select 10 random sorted row IDs
        let mut rng = rand::thread_rng();
        let mut row_ids: Vec<u64> = (0..10).map(|_| rng.gen_range(0..n as u64)).collect();
        row_ids.sort_unstable();
        row_ids.dedup();

        for &fmt in &[Parquet, Vortex] {
            let dir = tempfile::tempdir().unwrap();
            let filename = match fmt {
                Parquet => "data.parquet",
                Vortex => "data.vortex",
            };
            let path = dir.path().join(filename);

            // Write test data
            match fmt {
                Parquet => format::parquet::write_data(&path, schema.clone(), &batch).unwrap(),
                Vortex => format::vortex::write_data(&path, schema.clone(), &batch).unwrap(),
            }

            let label = format!("{fmt}/{n}x{dim}d/top10");
            group.bench_function(BenchmarkId::new(&label, n), |b| {
                b.iter(|| {
                    match fmt {
                        Parquet => {
                            format::parquet::read_metadata_by_row_ids(&path, &schema, &row_ids)
                                .unwrap()
                        }
                        Vortex => {
                            format::vortex::read_metadata_by_row_ids(&path, &schema, &row_ids)
                                .unwrap()
                        }
                    };
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_format_write,
    bench_format_read_all,
    bench_format_read_metadata
);
criterion_main!(benches);
