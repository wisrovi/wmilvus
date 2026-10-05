"""Pytest test setup and fixtures for WMilvus."""

import pytest
from wmilvus.types import VectorFieldConfig


@pytest.fixture
def sample_vector_config() -> VectorFieldConfig:
    """Fixture providing a standard vector field configuration for tests."""
    return VectorFieldConfig(dim=4, metric_type="COSINE", index_type="FLAT")
