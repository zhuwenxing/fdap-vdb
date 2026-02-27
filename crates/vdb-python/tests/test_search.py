import numpy as np

import fdap_vdb


def test_search_list(db_with_collection):
    db_with_collection.insert(
        "test",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadata={"category": ["tech", "sci"], "score": [42, 99]},
    )
    results = db_with_collection.search("test", query_vector=[1, 0, 0, 0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"
    assert results[0]["distance"] < results[1]["distance"]
    assert results[0]["metadata"]["category"] == "tech"


def test_search_numpy(db_with_collection):
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    db_with_collection.insert("test", ids=["a", "b"], vectors=vecs)

    query = np.array([1, 0, 0, 0], dtype=np.float32)
    results = db_with_collection.search("test", query_vector=query, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_search_default_top_k(db_with_collection):
    # Insert more than 10 items
    ids = [f"id{i}" for i in range(15)]
    vecs = [[float(i == j) for j in range(4)] for i in range(15)]
    # Fill remaining with random-ish vectors
    for i in range(4, 15):
        vecs[i] = [0.1 * i, 0.2 * i, 0.3, 0.1]
    db_with_collection.insert("test", ids=ids, vectors=vecs)

    # Default top_k should be 10
    results = db_with_collection.search("test", query_vector=[1, 0, 0, 0])
    assert len(results) == 10


def test_search_after_delete(db_with_collection):
    db_with_collection.insert(
        "test",
        ids=["a", "b", "c"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    )
    db_with_collection.delete("test", ids=["a"])
    results = db_with_collection.search("test", query_vector=[1, 0, 0, 0], top_k=10)
    assert all(r["id"] != "a" for r in results)


def test_search_dimension_mismatch(db_with_collection):
    db_with_collection.insert("test", ids=["a"], vectors=[[1, 0, 0, 0]])
    import pytest

    with pytest.raises(ValueError, match="dimension mismatch"):
        db_with_collection.search("test", query_vector=[1, 0, 0], top_k=1)


def test_search_nonexistent_collection(tmp_db):
    import pytest

    with pytest.raises(KeyError):
        tmp_db.search("nope", query_vector=[1, 0, 0, 0], top_k=1)
