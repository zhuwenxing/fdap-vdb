use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use vdb_common::config::{CollectionConfig, MetadataFieldConfig, MetadataFieldType};
use vdb_common::metrics::DistanceMetric;
use vdb_storage::engine::{MetadataColumnValue, StorageEngine};

fn create_engine(dir: &Path) -> Arc<StorageEngine> {
    Arc::new(StorageEngine::open(dir).unwrap())
}

fn test_collection_config(name: &str) -> CollectionConfig {
    CollectionConfig {
        name: name.to_string(),
        dimension: 4,
        distance_metric: DistanceMetric::Cosine,
        index_config: Default::default(),
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

#[test]
fn test_create_insert_search() {
    let dir = tempfile::tempdir().unwrap();
    let engine = create_engine(dir.path());

    // Create collection
    engine
        .create_collection(test_collection_config("docs"))
        .unwrap();

    // Insert vectors with metadata
    let n = 100;
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut categories = Vec::new();
    let mut scores = Vec::new();

    for i in 0..n {
        ids.push(format!("doc_{i}"));
        // Create vectors pointing in different directions
        let angle = (i as f32) * std::f32::consts::PI * 2.0 / (n as f32);
        vectors.push(vec![angle.cos(), angle.sin(), 0.0, 0.0]);
        categories.push(MetadataColumnValue::String(if i % 2 == 0 {
            "even".to_string()
        } else {
            "odd".to_string()
        }));
        scores.push(MetadataColumnValue::Int64(i as i64));
    }

    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), categories);
    metadata.insert("score".to_string(), scores);

    let count = engine.insert("docs", ids, vectors, metadata).unwrap();
    assert_eq!(count, n as u64);

    // Search: query vector pointing in direction of doc_0
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let hits = engine.search("docs", &query, 10, 0).unwrap();
    assert_eq!(hits.len(), 10);

    // Results should be sorted by distance (ascending)
    for i in 1..hits.len() {
        assert!(
            hits[i].distance >= hits[i - 1].distance,
            "results not sorted: {} >= {} at index {i}",
            hits[i].distance,
            hits[i - 1].distance,
        );
    }

    // Nearest should be doc_0 (same direction as query)
    assert_eq!(hits[0].id, "doc_0");
    assert!(
        hits[0].distance < 0.01,
        "nearest distance too large: {}",
        hits[0].distance
    );
}

#[test]
fn test_search_after_flush() {
    let dir = tempfile::tempdir().unwrap();
    let engine = create_engine(dir.path());

    engine
        .create_collection(test_collection_config("test_flush"))
        .unwrap();

    // Insert first batch
    let mut meta = HashMap::new();
    meta.insert(
        "category".to_string(),
        vec![MetadataColumnValue::String("first".to_string())],
    );
    meta.insert("score".to_string(), vec![MetadataColumnValue::Int64(1)]);
    engine
        .insert(
            "test_flush",
            vec!["v1".to_string()],
            vec![vec![1.0, 0.0, 0.0, 0.0]],
            meta,
        )
        .unwrap();

    // Flush to disk
    engine.flush("test_flush").unwrap();

    // Insert second batch (in memtable)
    let mut meta2 = HashMap::new();
    meta2.insert(
        "category".to_string(),
        vec![MetadataColumnValue::String("second".to_string())],
    );
    meta2.insert("score".to_string(), vec![MetadataColumnValue::Int64(2)]);
    engine
        .insert(
            "test_flush",
            vec!["v2".to_string()],
            vec![vec![0.0, 1.0, 0.0, 0.0]],
            meta2,
        )
        .unwrap();

    // Search should find both
    let hits = engine
        .search("test_flush", &[1.0, 0.0, 0.0, 0.0], 10, 0)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "v1");
}

#[test]
fn test_persistence() {
    let dir = tempfile::tempdir().unwrap();

    // Phase 1: create, insert, flush
    {
        let engine = create_engine(dir.path());
        engine
            .create_collection(test_collection_config("persist"))
            .unwrap();
        engine
            .insert(
                "persist",
                vec!["a".to_string(), "b".to_string()],
                vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
                HashMap::new(),
            )
            .unwrap();
        engine.flush("persist").unwrap();
    }

    // Phase 2: reopen and verify
    {
        let engine = create_engine(dir.path());
        let collections = engine.list_collections();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name, "persist");

        let hits = engine
            .search("persist", &[1.0, 0.0, 0.0, 0.0], 2, 0)
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a");
    }
}

#[test]
fn test_list_and_drop_collection() {
    let dir = tempfile::tempdir().unwrap();
    let engine = create_engine(dir.path());

    engine
        .create_collection(test_collection_config("col1"))
        .unwrap();
    engine
        .create_collection(test_collection_config("col2"))
        .unwrap();

    let list = engine.list_collections();
    assert_eq!(list.len(), 2);

    engine.drop_collection("col1").unwrap();
    let list = engine.list_collections();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "col2");
}
