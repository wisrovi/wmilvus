"""Comprehensive unit tests for WMilvus Pydantic ORM (Single, Multi-Collection, Batch, Range, Hybrid, Schema, and Async)."""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from wmilvus import AsyncWMilvus, FieldVector, ForensicModel, WMilvus


class Person(BaseModel):
    """Pydantic model for single-collection ORM testing."""

    id: str
    name: str
    age: int
    is_active: bool
    embedding: list[float] = FieldVector(dim=4, metric_type="COSINE")


class UserFace(ForensicModel):
    """Forensic model for multi-collection ORM testing."""

    __tablename__ = "users_face"
    id: str
    name: str
    face_vec: list[float] = FieldVector(dim=128)


class ProductImage(BaseModel):
    """Product image vector model."""

    id: str
    title: str
    price: float
    image_vec: list[float] = FieldVector(dim=64)


@patch("wmilvus.core.client.MilvusClient")
def test_single_collection_orm_flow(mock_milvus_client: MagicMock) -> None:
    """Test full single-collection ORM CRUD and similarity search returning Pydantic models."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = [
        {
            "id": "1",
            "name": "Juan Pérez",
            "age": 30,
            "is_active": True,
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.99,
                "entity": {
                    "id": "1",
                    "name": "Juan Pérez",
                    "age": 30,
                    "is_active": True,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                },
            }
        ]
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Insert Pydantic model
    person = Person(
        id="1", name="Juan Pérez", age=30, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]
    )
    inserted = db.insert(person)
    assert inserted.name == "Juan Pérez"
    mock_instance.upsert.assert_called()

    # 2. Query all & get_by_field returning Person instance
    people = db.get_all()
    assert len(people) == 1
    assert isinstance(people[0], Person)
    assert people[0].name == "Juan Pérez"

    single_person = db.get_by_field(name="Juan Pérez")
    assert single_person is not None
    assert single_person.age == 30

    # 3. Vector similarity search returning Person instance & with scores
    matches = db.search_similar(vector=[0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(matches) == 1
    assert isinstance(matches[0], Person)
    assert matches[0].name == "Juan Pérez"

    matches_scores = db.search_similar_with_scores(vector=[0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(matches_scores) == 1
    assert matches_scores[0].id == "1"
    assert matches_scores[0].distance == 0.99

    # 4. Update and Delete
    updated = db.update(
        "1",
        Person(id="1", name="Juan Pérez", age=31, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]),
    )
    assert updated.age == 31

    db.delete("1")
    mock_instance.delete.assert_called_with(collection_name="person", ids=["1"])


@patch("wmilvus.core.client.MilvusClient")
def test_ghost_table_audit_trail(mock_milvus_client: MagicMock) -> None:
    """Test ghost collection audit trail (_forensic_audit_log) tracking INSERT, UPDATE, and DELETE."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = [
        {
            "id": 1,
            "action_type": "INSERT",
            "table_name": "users_face",
            "record_id": "u1",
            "create_by": 100,
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(UserFace, config)

    user = UserFace(id="u1", name="William Rodriguez", face_vec=[0.1] * 128)
    db.insert(user, user_id=100)

    # Verify audit insert was called on _forensic_audit_log
    mock_instance.insert.assert_called()

    audit_logs = db.get_ghost_audit_log()
    assert len(audit_logs) == 1
    assert audit_logs[0]["action_type"] == "INSERT"
    assert audit_logs[0]["create_by"] == 100


@patch("wmilvus.core.client.MilvusClient")
def test_multi_collection_orm_routing(mock_milvus_client: MagicMock) -> None:
    """Test multi-collection routing via class indexing, attribute access, and auto-routing."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = []
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus([UserFace, ProductImage], config)

    user = UserFace(id="u1", name="William Rodriguez", face_vec=[0.1] * 128)
    prod = ProductImage(id="p101", title="Camiseta Vision", price=29.99, image_vec=[0.5] * 64)

    # Method A: Indexing by class (Type-safe)
    db[UserFace].insert(user)

    # Method B: Direct attribute access in lowercase
    db.productimage.insert(prod)

    # Method C: Auto-routing insert
    db.insert(user)

    assert mock_instance.upsert.call_count >= 3


@patch("wmilvus.core.client.MilvusClient")
def test_batch_ingestion_and_range_search(mock_milvus_client: MagicMock) -> None:
    """Test insert_batch, search_range_with_scores, search_hybrid, and verify_schema."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.describe_collection.return_value = {
        "fields": [{"name": "vector", "params": {"dim": 4}}]
    }
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.96,
                "entity": {
                    "id": "1",
                    "name": "Juan Pérez",
                    "age": 30,
                    "is_active": True,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                },
            }
        ]
    ]
    mock_instance.query.return_value = [
        {
            "id": "1",
            "name": "Juan Pérez",
            "age": 30,
            "is_active": True,
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Test insert_batch
    persons = [
        Person(
            id=f"{i}", name=f"User_{i}", age=20 + i, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]
        )
        for i in range(5)
    ]
    batch_res = db.insert_batch(persons, batch_size=2)
    assert batch_res["inserted_count"] == 5
    assert batch_res["batches_processed"] == 3

    # 2. Test search_range_with_scores
    range_matches = db.search_range_with_scores(vector=[0.1, 0.2, 0.3, 0.4], radius=0.90)
    assert len(range_matches) == 1
    assert range_matches[0].distance == 0.96

    # 3. Test search_hybrid
    hybrid_res = db.search_hybrid(vector=[0.1, 0.2, 0.3, 0.4], text_field="name", text_query="Juan")
    assert len(hybrid_res) == 1

    # 4. Test verify_schema
    schema_res = db.verify_schema()
    assert schema_res["status"] == "VALID"
    assert schema_res["dimension"] == 4


@patch("wmilvus.core.client.MilvusClient")
def test_async_client_flow(mock_milvus_client: MagicMock) -> None:
    """Test AsyncWMilvus client operations using async/await."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = [
        {
            "id": "1",
            "name": "Juan Pérez",
            "age": 30,
            "is_active": True,
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.99,
                "entity": {
                    "id": "1",
                    "name": "Juan Pérez",
                    "age": 30,
                    "is_active": True,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                },
            }
        ]
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}

    async def _run() -> None:
        async with AsyncWMilvus(Person, config) as db:
            person = Person(
                id="1", name="Juan Pérez", age=30, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]
            )
            inserted = await db.insert(person)
            assert inserted.name == "Juan Pérez"

            matches = await db.search_similar(vector=[0.1, 0.2, 0.3, 0.4], top_k=1)
            assert len(matches) == 1
            assert matches[0].name == "Juan Pérez"

    import asyncio

    asyncio.run(_run())


def test_normalize_vector() -> None:
    """Test L2 vector normalization utility."""
    from wmilvus.types import normalize_vector

    vec = [3.0, 4.0]
    norm = normalize_vector(vec)
    assert abs(norm[0] - 0.6) < 1e-5
    assert abs(norm[1] - 0.8) < 1e-5

    # Zero vector case
    zero_vec = [0.0, 0.0]
    assert normalize_vector(zero_vec) == [0.0, 0.0]


@patch("wmilvus.core.client.MilvusClient")
def test_wpipe_ingest_step(mock_milvus_client: MagicMock) -> None:
    """Test WMilvusIngestStep integration for WPipe pipeline execution."""
    from wmilvus.integrations.wpipe import WMilvusIngestStep

    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    step = WMilvusIngestStep(Person, config)

    person = Person(
        id="w1", name="Pipe User", age=28, is_active=True, embedding=[0.5, 0.5, 0.5, 0.5]
    )
    result = step.process([person])
    assert result["status"] == "SUCCESS"
    assert result["inserted_count"] == 1
