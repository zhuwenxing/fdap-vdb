import tempfile

import pytest

import fdap_vdb


@pytest.fixture
def tmp_db(tmp_path):
    """Create a VDB instance with a temporary data directory."""
    return fdap_vdb.VDB(str(tmp_path))


@pytest.fixture
def db_with_collection(tmp_db):
    """Create a VDB instance with a pre-created collection (dim=4)."""
    tmp_db.create_collection(
        "test",
        dimension=4,
        metadata_fields=[("category", "string"), ("score", "int64")],
    )
    return tmp_db
