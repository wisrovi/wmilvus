"""Unit tests for WMilvus client, models, and operations."""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from wmilvus import SearchMatch, VectorRecord, WMilvus


def test_vector_record_creation(sample_vector_record: VectorRecord) -> None:
    """Validate construction and field types of VectorRecord."""
    assert sample_vector_record.id == "test_vec_1"
    assert len(sample_vector_record.vector) == 4
    assert sample_vector_record.metadata["category"] == "test"


def test_search_match_model() -> None:
    """Validate construction and attributes of SearchMatch."""
    match = SearchMatch(id="match_123", distance=0.985, metadata={"label": "person"})
    assert match.id == "match_123"
    assert match.distance == 0.985
    assert match.metadata["label"] == "person"


@patch("wmilvus.core.client.MilvusClient")
def test_ensure_collection(mock_milvus_client: MagicMock) -> None:
    """Test ensure_collection creates collection if it does not exist."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    client.ensure_collection(collection_name="test_col", dimension=128, metric_type="COSINE")

    mock_instance.has_collection.assert_called_once_with(collection_name="test_col")
    mock_instance.create_collection.assert_called_once()


@patch("wmilvus.core.client.MilvusClient")
def test_upsert_batch(mock_milvus_client: MagicMock, sample_vector_record: VectorRecord) -> None:
    """Test upsert_batch converts VectorRecords to payload dictionaries."""
    mock_instance = MagicMock()
    mock_instance.upsert.return_value = {"upsert_count": 1}
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    result = client.upsert_batch(collection_name="test_col", records=[sample_vector_record])

    assert result["upsert_count"] == 1
    mock_instance.upsert.assert_called_once_with(
        collection_name="test_col",
        data=[
            {
                "id": "test_vec_1",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "category": "test",
                "active": True,
            }
        ],
    )


@patch("wmilvus.core.client.MilvusClient")
def test_search_similar(mock_milvus_client: MagicMock) -> None:
    """Test search_similar parses NumPy query vectors and formats SearchMatch results."""
    mock_instance = MagicMock()
    mock_instance.search.return_value = [
        [
            {
                "id": "vec_001",
                "distance": 0.95,
                "entity": {"id": "vec_001", "label": "forklift"},
            }
        ]
    ]
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    query_arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    matches = client.search_similar(
        collection_name="test_col",
        query_vector=query_arr,
        top_k=1,
        filter_expr='label == "forklift"',
    )

    assert len(matches) == 1
    assert matches[0].id == "vec_001"
    assert matches[0].distance == 0.95
    assert matches[0].metadata == {"label": "forklift"}
