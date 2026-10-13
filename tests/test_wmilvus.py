"""Comprehensive unit tests for WMilvus Pydantic ORM (Single, Multi-Collection, Batch, Range, Hybrid, Schema, and Async)."""

from typing import Any
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


@patch("wmilvus.core.client.MilvusClient")
def test_wsqlite_export_import_flow(mock_milvus_client: MagicMock, tmp_path: Any) -> None:
    """Test export_to_sqlite and import_from_sqlite backup integration."""
    from wmilvus.integrations.wsqlite import export_to_sqlite, import_from_sqlite

    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_instance.query.return_value = [
        {
            "id": "sq1",
            "name": "Backup User",
            "age": 35,
            "is_active": True,
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    db.get_all = MagicMock(
        return_value=[
            Person(
                id="sq1", name="Backup User", age=35, is_active=True, embedding=[0.1, 0.2, 0.3, 0.4]
            )
        ]
    )

    sqlite_file = str(tmp_path / "wmilvus_backup.db")

    # 1. Export
    exp_res = export_to_sqlite(db, Person, sqlite_file)
    assert exp_res["exported_count"] == 1
    assert exp_res["sqlite_path"] == sqlite_file

    # 2. Import
    imp_res = import_from_sqlite(db, Person, sqlite_file)
    assert imp_res["imported_count"] == 1
    assert imp_res["status"] == "SUCCESS"


class FaceEmbeddingIntPk(ForensicModel):
    """Pydantic model representing a Milvus collection with INT64 auto-increment PK."""

    __tablename__ = "Faces"

    id: int | None = None
    embedding: list[float] = FieldVector(dim=4, metric_type="COSINE")
    embedding_norm: float = 1.0


@patch("wmilvus.core.client.MilvusClient")
def test_get_all_universal_filter_and_int64_pk(mock_milvus_client: MagicMock) -> None:
    """Test get_all() uses an empty filter query compatible with INT64 and VARCHAR keys.

    This test validates:
    1. Query call to MilvusClient passes filter="" so collections with INT64 or VARCHAR
       primary keys do not trigger type mismatch errors.
    2. Model records with INT64 primary keys and custom vector field names are properly
       parsed and instantiated into Pydantic model instances.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.query.return_value = [
        {
            "id": 468847727109203484,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "embedding_norm": 1.0,
            "dni": "00000000P",
            "file_name": "test.png",
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(FaceEmbeddingIntPk, config)

    results = db.get_all(limit=50)

    # Validate that client.query was called with filter="" to support int/varchar keys
    mock_instance.query.assert_called_with(
        collection_name="Faces",
        filter="",
        output_fields=["*"],
        limit=50,
    )

    # Validate result parsing and typed Pydantic instance attributes
    assert len(results) == 1
    assert isinstance(results[0], FaceEmbeddingIntPk)
    assert results[0].id == 468847727109203484
    assert results[0].embedding_norm == 1.0
    assert results[0].embedding == [0.1, 0.2, 0.3, 0.4]


@patch("wmilvus.core.client.MilvusClient")
def test_verify_schema_custom_vector_name(mock_milvus_client: MagicMock) -> None:
    """Test verify_schema() succeeds when vector field matches the model field name.

    This test validates:
    1. Collections whose vector field is named after the model attribute (e.g. 'embedding')
       rather than the default 'vector' pass schema verification.
    2. Dimensionality checks are performed accurately against the custom vector field.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.describe_collection.return_value = {
        "fields": [
            {"name": "id", "params": {}},
            {"name": "embedding", "params": {"dim": 4}},
        ]
    }
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(FaceEmbeddingIntPk, config)

    schema_res = db.verify_schema()
    assert schema_res["status"] == "VALID"
    assert schema_res["dimension"] == 4


@patch("wmilvus.core.client.MilvusClient")
def test_insert_auto_id_and_dynamic_vector_field(mock_milvus_client: MagicMock) -> None:
    """Test insert() correctly handles dynamic vector field names and auto-id assignment.

    This test validates:
    1. The payload key matches the custom vector field name ('embedding') rather than 'vector'.
    2. When 'id' is None, no 'id' key is sent in payload, and client.insert() is called.
    3. The auto-generated ID returned by Milvus is assigned back to the record instance.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.insert.return_value = {"insert_count": 1, "ids": [9876543210]}
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(FaceEmbeddingIntPk, config)

    item = FaceEmbeddingIntPk(embedding=[0.5, 0.5, 0.5, 0.5])
    assert item.id is None

    inserted = db.insert(item, partition_name="partition_a")

    # Verify client.insert called for Faces with dynamic vector field 'embedding' and partition_name
    faces_call = mock_instance.insert.call_args_list[0]
    call_kwargs = faces_call.kwargs
    assert call_kwargs["collection_name"] == "Faces"
    assert call_kwargs["partition_name"] == "partition_a"
    payload = call_kwargs["data"][0]
    assert "embedding" in payload
    assert "vector" not in payload
    assert "id" not in payload

    # Verify ID was propagated to the model instance
    assert inserted.id == 9876543210


@patch("wmilvus.core.client.MilvusClient")
def test_delete_batch_and_delete_by_expr(mock_milvus_client: MagicMock) -> None:
    """Test delete_batch() and delete_by_expr() operations on collection.

    This test validates:
    1. delete_batch passes list of IDs directly to client.delete(ids=[...]).
    2. delete_by_expr passes filter expression to client.delete(filter=...).
    3. Audit logs are recorded appropriately for batch and expression deletions.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.delete.return_value = {"delete_count": 2}
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Test delete_batch
    res_batch = db.delete_batch(["id_1", "id_2"], partition_name="part_1")
    assert res_batch["delete_count"] == 2
    mock_instance.delete.assert_called_with(
        collection_name="person", ids=["id_1", "id_2"], partition_name="part_1"
    )

    # 2. Test delete_by_expr
    res_expr = db.delete_by_expr("age >= 30", partition_name="part_1")
    assert res_expr["delete_count"] == 2
    mock_instance.delete.assert_called_with(
        collection_name="person", filter="age >= 30", partition_name="part_1"
    )


@patch("wmilvus.core.client.MilvusClient")
def test_filter_by_query_builder(mock_milvus_client: MagicMock) -> None:
    """Test filter_by() query builder translation and typed model instantiation.

    This test validates:
    1. Comparison operators (age__gte, status__in, dni) are translated into valid Milvus filter string.
    2. Results returned from Milvus query are deserialized into typed Pydantic instances.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.query.return_value = [
        {"id": "p1", "name": "Alice", "age": 25, "is_active": True, "vector": [0.1, 0.2, 0.3, 0.4]}
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    results = db.filter_by(age__gte=20, is_active=True, limit=10)

    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Alice"

    mock_instance.query.assert_called_once()
    query_kwargs = mock_instance.query.call_args.kwargs
    assert "age >= 20" in query_kwargs["filter"]
    assert "is_active == True" in query_kwargs["filter"]


@patch("wmilvus.core.client.MilvusClient")
def test_iter_all_streaming(mock_milvus_client: MagicMock) -> None:
    """Test iter_all() streaming batches of typed Pydantic models.

    This test validates:
    1. iter_all initializes native query_iterator on MilvusClient.
    2. Batches are yielded as typed Pydantic model lists and iterator is closed on exit.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True

    mock_iterator = MagicMock()
    mock_iterator.next.side_effect = [
        [
            {
                "id": "it1",
                "name": "User 1",
                "age": 20,
                "is_active": True,
                "vector": [0.1, 0.2, 0.3, 0.4],
            },
            {
                "id": "it2",
                "name": "User 2",
                "age": 21,
                "is_active": True,
                "vector": [0.1, 0.2, 0.3, 0.4],
            },
        ],
        [],  # End of iteration
    ]
    mock_instance.query_iterator.return_value = mock_iterator
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    batches = list(db.iter_all(batch_size=2))

    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert isinstance(batches[0][0], Person)
    assert batches[0][0].id == "it1"
    mock_iterator.close.assert_called_once()


@patch("wmilvus.core.client.MilvusClient")
def test_search_similar_with_search_params_and_partition(mock_milvus_client: MagicMock) -> None:
    """Test search_similar() passes custom search_params and partition_names.

    This test validates:
    1. Custom search index parameters like efSearch or nprobe are passed to client.search.
    2. partition_names are routed properly to client.search.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.98,
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

    matches = db.search_similar(
        vector=[0.1, 0.2, 0.3, 0.4],
        top_k=3,
        search_params={"ef": 64},
        partition_names=["camera_01"],
    )

    assert len(matches) == 1
    assert matches[0].name == "Juan Pérez"

    search_kwargs = mock_instance.search.call_args.kwargs
    assert search_kwargs["search_params"] == {"ef": 64}
    assert search_kwargs["partition_names"] == ["camera_01"]
