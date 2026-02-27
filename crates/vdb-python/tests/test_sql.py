import pyarrow as pa

import fdap_vdb


def test_sql_select(db_with_collection):
    db_with_collection.insert(
        "test",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadata={"category": ["tech", "sci"], "score": [42, 99]},
    )
    batch = db_with_collection.sql("SELECT _id, category, score FROM test ORDER BY _id")
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 2
    assert batch.column("_id").to_pylist() == ["a", "b"]
    assert batch.column("category").to_pylist() == ["tech", "sci"]
    assert batch.column("score").to_pylist() == [42, 99]


def test_sql_where(db_with_collection):
    db_with_collection.insert(
        "test",
        ids=["a", "b", "c"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        metadata={"category": ["tech", "sci", "tech"], "score": [42, 99, 10]},
    )
    batch = db_with_collection.sql("SELECT _id FROM test WHERE score > 30 ORDER BY _id")
    assert batch.num_rows == 2
    assert batch.column("_id").to_pylist() == ["a", "b"]


def test_sql_count(db_with_collection):
    db_with_collection.insert(
        "test",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
    )
    batch = db_with_collection.sql("SELECT COUNT(*) AS cnt FROM test")
    assert batch.num_rows == 1
    assert batch.column("cnt").to_pylist() == [2]


def test_sql_returns_pyarrow_batch(db_with_collection):
    db_with_collection.insert("test", ids=["a"], vectors=[[1, 0, 0, 0]])
    batch = db_with_collection.sql("SELECT * FROM test")
    assert isinstance(batch, pa.RecordBatch)
    assert "_id" in batch.schema.names
