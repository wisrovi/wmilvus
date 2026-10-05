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
def test_context_manager_and_close(mock_milvus_client: MagicMock) -> None:
    """Test context manager (__enter__, __exit__) and explicit close."""
    mock_instance = MagicMock()
    mock_milvus_client.return_value = mock_instance

    with WMilvus(uri="http://localhost:19530") as client:
        assert client.uri == "http://localhost:19530"

    mock_instance.close.assert_called_once()


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
def test_upsert_batch_with_chunking(mock_milvus_client: MagicMock, sample_vector_record: VectorRecord) -> None:
    """Test upsert_batch with chunking splits large payloads correctly."""
    mock_instance = MagicMock()
    mock_instance.upsert.return_value = {"upsert_count": 1}
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    records = [sample_vector_record, sample_vector_record]
    result = client.upsert_batch(collection_name="test_col", records=records, chunk_size=1)

    assert result["upsert_count"] == 2
    assert mock_instance.upsert.call_count == 2


@patch("wmilvus.core.client.MilvusClient")
def test_search_batch(mock_milvus_client: MagicMock) -> None:
    """Test search_batch handles multiple query vectors in parallel."""
    mock_instance = MagicMock()
    mock_instance.search.return_value = [
        [{"id": "vec_001", "distance": 0.95, "entity": {"id": "vec_001", "label": "forklift"}}],
        [{"id": "vec_002", "distance": 0.88, "entity": {"id": "vec_002", "label": "truck"}}],
    ]
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    queries = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    batch_results = client.search_batch(collection_name="test_col", query_vectors=queries, top_k=1)

    assert len(batch_results) == 2
    assert batch_results[0][0].id == "vec_001"
    assert batch_results[1][0].id == "vec_002"


@patch("wmilvus.core.client.MilvusClient")
def test_query_scalar(mock_milvus_client: MagicMock) -> None:
    """Test query_scalar executes scalar filtering queries."""
    mock_instance = MagicMock()
    mock_instance.query.return_value = [{"id": "vec_001", "camera_id": "cam_north"}]
    mock_milvus_client.return_value = mock_instance

    client = WMilvus(uri="http://localhost:19530")
    res = client.query_scalar(collection_name="test_col", filter_expr='camera_id == "cam_north"')

    assert len(res) == 1
    assert res[0]["camera_id"] == "cam_north"
    mock_instance.query.assert_called_once_with(
        collection_name="test_col",
        filter='camera_id == "cam_north"',
        output_fields=["*"],
        limit=100,
    )
