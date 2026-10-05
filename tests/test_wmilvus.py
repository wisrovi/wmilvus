"""Unit tests for WMilvus core classes, exceptions, and configuration validation."""

import pytest
from pydantic import BaseModel

from wmilvus import WMilvus, ConnectionManager, VectorFieldConfig, SearchResult
from wmilvus.exceptions import ConnectionError, WMilvusError


class FaceEmbedding(BaseModel):
    """Sample Pydantic model for unit testing vector schema mapping."""
    user_id: str
    feature_name: str


def test_vector_field_config_defaults() -> None:
    """Validate default parameter values for VectorFieldConfig."""
    config = VectorFieldConfig(dim=512)
    assert config.dim == 512
    assert config.name == "vector"
    assert config.metric_type == "COSINE"
    assert config.index_type == "HNSW"
    assert config.params["M"] == 16


def test_search_result_model() -> None:
    """Validate construction and attribute access of SearchResult model."""
    result = SearchResult(id=101, distance=0.985, entity={"name": "test"})
    assert result.id == 101
    assert result.distance == 0.985
    assert result.entity["name"] == "test"


def test_connection_manager_initialization() -> None:
    """Ensure ConnectionManager sets up parameters correctly without connecting immediately."""
    conn = ConnectionManager(host="127.0.0.1", port=19530, alias="test_alias")
    assert conn.host == "127.0.0.1"
    assert conn.port == 19530
    assert conn.alias == "test_alias"
    assert not conn.is_connected


def test_wmilvus_client_instantiation(sample_vector_config: VectorFieldConfig) -> None:
    """Test instantiating WMilvus client with a Pydantic model class."""
    client = WMilvus(
        model_class=FaceEmbedding,
        vector_config=sample_vector_config,
        host="localhost",
        port=19530,
    )
    assert client.collection_name == "faceembedding"
    assert client.vector_config.dim == 4
    assert client.connection.host == "localhost"
