"""Pytest test setup and fixtures for WMilvus."""

import pytest
from wmilvus.types import VectorRecord


@pytest.fixture
def sample_vector_record() -> VectorRecord:
    """Fixture providing a sample VectorRecord instance for unit tests."""
    return VectorRecord(
        id="test_vec_1",
        vector=[0.1, 0.2, 0.3, 0.4],
        metadata={"category": "test", "active": True},
    )
