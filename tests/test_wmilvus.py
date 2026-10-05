"""Comprehensive unit tests for WMilvus Pydantic ORM (Single, Multi-Collection, and Ghost Audit Log)."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from wmilvus import FieldVector, ForensicModel, WMilvus


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
        {"id": "1", "name": "Juan Pérez", "age": 30, "is_active": True, "vector": [0.1, 0.2, 0.3, 0.4]}
    ]
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.99,
                "entity": {"id": "1", "name": "Juan Pérez", "age": 30, "is_active": True, "vector": [0.1, 0.2, 0.3, 0.4]},
            }
        ]
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Insert Pydantic model
    person = Person(id="1", name="Juan Pérez", age=30, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4])
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
    updated = db.update("1", Person(id="1", name="Juan Pérez", age=31, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]))
    assert updated.age == 31

    db.delete("1")
    mock_instance.delete.assert_called_with(collection_name="person", ids=["1"])


@patch("wmilvus.core.client.MilvusClient")
def test_ghost_table_audit_trail(mock_milvus_client: MagicMock) -> None:
    """Test ghost collection audit trail (_forensic_audit_log) tracking INSERT, UPDATE, and DELETE."""
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = [
        {"id": 1, "action_type": "INSERT", "table_name": "users_face", "record_id": "u1", "create_by": 100}
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
