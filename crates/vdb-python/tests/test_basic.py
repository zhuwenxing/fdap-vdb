import pytest

import fdap_vdb


def test_create_and_list(tmp_db):
    tmp_db.create_collection("demo", dimension=8)
    cols = tmp_db.list_collections()
    assert len(cols) == 1
    assert cols[0]["name"] == "demo"
    assert cols[0]["dimension"] == 8
    assert cols[0]["distance_metric"] == "cosine"


def test_create_with_metric(tmp_db):
    tmp_db.create_collection("demo", dimension=4, distance_metric="l2")
    cols = tmp_db.list_collections()
    assert cols[0]["distance_metric"] == "l2"


def test_create_with_metadata_fields(tmp_db):
    tmp_db.create_collection(
        "demo",
        dimension=4,
        metadata_fields=[("category", "string"), ("score", "int64")],
    )
    cols = tmp_db.list_collections()
    fields = cols[0]["metadata_fields"]
    assert ("category", "string") in fields
    assert ("score", "int64") in fields


def test_create_duplicate_raises(tmp_db):
    tmp_db.create_collection("demo", dimension=4)
    with pytest.raises(ValueError):
        tmp_db.create_collection("demo", dimension=4)


def test_drop_collection(tmp_db):
    tmp_db.create_collection("demo", dimension=4)
    tmp_db.drop_collection("demo")
    assert len(tmp_db.list_collections()) == 0


def test_drop_nonexistent_raises(tmp_db):
    with pytest.raises(KeyError):
        tmp_db.drop_collection("nonexistent")


def test_insert_list(db_with_collection):
    n = db_with_collection.insert(
        "test",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadata={"category": ["tech", "sci"], "score": [42, 99]},
    )
    assert n == 2


def test_insert_dimension_mismatch(db_with_collection):
    with pytest.raises(ValueError, match="dimension mismatch"):
        db_with_collection.insert(
            "test",
            ids=["a"],
            vectors=[[1, 0, 0]],  # dim=3, expected 4
        )


def test_insert_ids_vectors_length_mismatch(db_with_collection):
    with pytest.raises(ValueError, match="ids length"):
        db_with_collection.insert(
            "test",
            ids=["a", "b"],
            vectors=[[1, 0, 0, 0]],  # 1 vector for 2 ids
        )


def test_delete(db_with_collection):
    db_with_collection.insert(
        "test", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]]
    )
    count = db_with_collection.delete("test", ids=["a"])
    assert count == 1

    # Delete same ID again returns 0
    count = db_with_collection.delete("test", ids=["a"])
    assert count == 0


def test_flush(db_with_collection):
    db_with_collection.insert("test", ids=["a"], vectors=[[1, 0, 0, 0]])
    db_with_collection.flush("test")
    # After flush, data should still be searchable
    results = db_with_collection.search("test", query_vector=[1, 0, 0, 0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_compact(db_with_collection):
    db_with_collection.insert("test", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]])
    db_with_collection.flush("test")
    db_with_collection.insert("test", ids=["c"], vectors=[[0, 0, 1, 0]])
    db_with_collection.flush("test")

    db_with_collection.delete("test", ids=["a"])

    result = db_with_collection.compact("test")
    assert result["segments_before"] == 2
    assert result["segments_after"] == 1
    assert result["rows_removed"] == 1
