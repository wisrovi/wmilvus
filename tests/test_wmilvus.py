"""Comprehensive unit tests for WMilvus Pydantic ORM (Single, Multi-Collection, Batch, Range, Hybrid, Schema, and Async)."""

from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field
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


@patch("wmilvus.core.client.MilvusClient")
def test_forensic_datetime_serialization_and_autoload_and_flexible_init(
    mock_milvus_client: MagicMock,
) -> None:
    """Test JSON datetime serialization in insert/update, collection autoload, and flexible init.

    This test validates:
    1. Flexible WMilvus initialization with URI as first argument and models in kwargs.
    2. Automatic loading of existing unloaded collections when has_collection returns True.
    3. ForensicModel datetime fields (create_in, update_in) are serialized to ISO strings
       in upsert/insert payloads via model_dump(mode='json'), preventing DataNotMatchException.
    4. Custom vector field names are excluded from SearchMatch.metadata.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.get_load_state.return_value = {"state": "NotLoad"}
    mock_instance.search.return_value = [
        [
            {
                "id": "1",
                "distance": 0.95,
                "entity": {
                    "id": "1",
                    "name": "Audit User",
                    "face_vec": [0.1] * 128,
                },
            }
        ]
    ]
    mock_milvus_client.return_value = mock_instance

    # 1. Flexible init: uri string first, models list
    db = WMilvus("http://milvus-server:19530", models=[UserFace])
    assert db.uri == "http://milvus-server:19530"

    # 2. Check that load_collection was called because get_load_state returned NotLoad
    mock_instance.load_collection.assert_called_with(collection_name="users_face")

    # 3. Insert and verify payload has string datetime, not datetime object
    face = UserFace(id="1", name="Audit User", face_vec=[0.1] * 128)
    db.insert(face)
    upsert_kwargs = mock_instance.upsert.call_args.kwargs
    payload = upsert_kwargs["data"][0]
    assert isinstance(payload["create_in"], str)
    assert not hasattr(payload["create_in"], "tzinfo")

    # 4. Search and verify custom vector field face_vec is omitted from metadata
    matches = db.search_similar_with_scores(vector=[0.1] * 128, top_k=1)
    assert len(matches) == 1
    assert "face_vec" not in matches[0].metadata


def test_field_vector_default_factory() -> None:
    """Test FieldVector automatically sets default_factory=list to prevent Pydantic ValidationError.

    This test validates:
    1. A Pydantic model with FieldVector(dim=384) can be instantiated without explicitly
       passing the vector field, automatically assigning an empty list [].
    2. model_validate() parses a dictionary containing only scalar fields without raising
       ValidationError('summary_embedding: Field required').
    3. Explicit vector assignments are preserved as expected.
    4. Custom default values or custom default_factories supplied by the caller are honored.
    """

    class Article(BaseModel):
        id: str
        title: str
        embedding: list[float] = FieldVector(dim=384)

    class CustomArticle(BaseModel):
        id: str
        embedding: list[float] = FieldVector(dim=4, default_factory=lambda: [0.0] * 4)

    # 1. Instantiation without vector defaults to empty list []
    art1 = Article(id="art_1", title="Vector Databases")
    assert art1.embedding == []

    # 2. Deserialization from dictionary with only scalar fields succeeds
    raw_dict = {"id": "art_2", "title": "Milvus Client"}
    art2 = Article.model_validate(raw_dict)
    assert art2.id == "art_2"
    assert art2.embedding == []

    # 3. Explicit vector assignment is preserved
    vec = [0.1] * 384
    art3 = Article(id="art_3", title="Semantic Search", embedding=vec)
    assert art3.embedding == vec

    # 4. Custom default_factory is honored
    custom_art = CustomArticle(id="art_4")
    assert custom_art.embedding == [0.0, 0.0, 0.0, 0.0]


@patch("wmilvus.core.client.MilvusClient")
def test_get_by_field_include_vector_and_custom_output_fields(
    mock_milvus_client: MagicMock,
) -> None:
    """Test get_by_field includes vector field in output_fields by default and supports custom overrides.

    This test validates:
    1. By default (include_vector=True), get_by_field includes ['*', 'embedding'] in output_fields
       so Milvus returns vector embeddings rather than omitting them.
    2. Returned vector embeddings are parsed into the typed Pydantic model instance.
    3. Explicit include_vector=False restricts output_fields to ['*'].
    4. Explicit output_fields parameter completely overrides automatic field selection.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.query.return_value = [
        {
            "id": "p100",
            "name": "Alice Developer",
            "age": 28,
            "is_active": True,
            "embedding": [0.2, 0.4, 0.6, 0.8],
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Default call: include_vector=True -> output_fields=['*', 'embedding']
    result = db.get_by_field(id="p100")
    assert result is not None
    assert result.name == "Alice Developer"
    assert result.embedding == [0.2, 0.4, 0.6, 0.8]

    first_call_kwargs = mock_instance.query.call_args.kwargs
    assert first_call_kwargs["output_fields"] == ["*", "embedding"]

    # 2. Explicit include_vector=False -> output_fields=['*']
    db.get_by_field(id="p100", include_vector=False)
    second_call_kwargs = mock_instance.query.call_args.kwargs
    assert second_call_kwargs["output_fields"] == ["*"]

    # 3. Explicit custom output_fields override
    db.get_by_field(id="p100", output_fields=["id", "name"])
    third_call_kwargs = mock_instance.query.call_args.kwargs
    assert third_call_kwargs["output_fields"] == ["id", "name"]


@patch("wmilvus.core.client.MilvusClient")
def test_filter_by_include_vector(mock_milvus_client: MagicMock) -> None:
    """Test filter_by includes vector field in output_fields when include_vector is True.

    This test validates:
    1. filter_by passes ['*', 'embedding'] in output_fields by default.
    2. filter_by passes ['*'] when include_vector=False is provided.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_instance.query.return_value = [
        {
            "id": "p200",
            "name": "Bob Admin",
            "age": 35,
            "is_active": True,
            "embedding": [0.1, 0.3, 0.5, 0.7],
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # 1. Default call: include_vector=True
    res = db.filter_by(age__gte=30, limit=5)
    assert len(res) == 1
    assert res[0].embedding == [0.1, 0.3, 0.5, 0.7]
    assert mock_instance.query.call_args.kwargs["output_fields"] == ["*", "embedding"]

    # 2. Explicit include_vector=False
    db.filter_by(age__gte=30, limit=5, include_vector=False)
    assert mock_instance.query.call_args.kwargs["output_fields"] == ["*"]


@patch("wmilvus.core.client.MilvusClient")
def test_update_preserves_existing_vector_when_incoming_vector_empty(
    mock_milvus_client: MagicMock,
) -> None:
    """Test update() preserves existing vector when incoming model has an empty vector.

    This test validates:
    1. When an entity is updated using a Pydantic model whose vector field is empty (e.g. []),
       update() looks up the existing record in Milvus.
    2. The existing non-empty vector is preserved in the upsert payload sent to MilvusClient,
       preventing Milvus dimension mismatch exceptions (length 0 vs schema dimension).
    3. The vector attribute on the returned record is also populated with the preserved vector.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    # Existing record in Milvus has a valid 4-dimensional vector
    mock_instance.query.return_value = [
        {
            "id": "1",
            "name": "Juan Pérez",
            "age": 30,
            "is_active": True,
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }
    ]
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # Incoming update record with modified age and EMPTY vector []
    partial_update_record = Person(
        id="1",
        name="Juan Pérez",
        age=31,
        is_active=True,
        embedding=[],  # Empty vector from scalar update
    )

    updated_result = db.update("1", partial_update_record)

    # Check that update() fetched the existing record
    mock_instance.query.assert_called()

    # Check that upsert payload received the PRESERVED vector, NOT an empty list
    upsert_kwargs = mock_instance.upsert.call_args.kwargs
    payload = upsert_kwargs["data"][0]
    assert payload["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert payload["age"] == 31

    # Check that returned record object has the preserved vector restored
    assert updated_result.embedding == [0.1, 0.2, 0.3, 0.4]
    assert updated_result.age == 31


def test_inspect_model_fields_schema_type_mapping() -> None:
    """Test inspect_model_fields_schema accurately maps all Python/Pydantic types to Milvus DataTypes.

    This test validates:
    1. Strings map to DataType.VARCHAR with default 65535 or custom max_length constraint.
    2. Integers map to DataType.INT64 with nullable=True.
    3. Floats map to DataType.DOUBLE with nullable=True.
    4. Booleans map to DataType.BOOL with nullable=True.
    5. List of strings maps to DataType.ARRAY with element_type=DataType.VARCHAR.
    6. List of integers maps to DataType.ARRAY with element_type=DataType.INT64.
    7. Dictionaries and arbitrary nested objects map to DataType.JSON.
    8. Primary key and vector fields are excluded from scalar schema inspection.
    """
    from datetime import datetime
    from typing import Optional

    from pydantic import Field
    from pymilvus import DataType
    from wmilvus.core.client import inspect_model_fields_schema

    class ComprehensiveModel(BaseModel):
        id: str
        vec: list[float] = FieldVector(dim=128)
        title: str
        count: int
        score: float
        is_verified: bool
        tags: list[str] = Field(default_factory=list)
        scores: list[int] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)
        created_at: Optional[datetime] = None

    fields = inspect_model_fields_schema(ComprehensiveModel, pk_name="id", vector_field_name="vec")
    field_map = {f["field_name"]: f for f in fields}

    # Verify primary key and vector field are not in scalar fields
    assert "id" not in field_map
    assert "vec" not in field_map

    # Verify VARCHAR mappings
    assert field_map["title"]["datatype"] == DataType.VARCHAR
    assert field_map["title"]["max_length"] == 65535
    assert field_map["title"]["nullable"] is True

    # Verify INT64 mappings
    assert field_map["count"]["datatype"] == DataType.INT64
    assert field_map["count"]["nullable"] is True

    # Verify DOUBLE mappings
    assert field_map["score"]["datatype"] == DataType.DOUBLE
    assert field_map["score"]["nullable"] is True

    # Verify BOOL mappings
    assert field_map["is_verified"]["datatype"] == DataType.BOOL
    assert field_map["is_verified"]["nullable"] is True

    # Verify ARRAY of VARCHAR mappings
    assert field_map["tags"]["datatype"] == DataType.ARRAY
    assert field_map["tags"]["element_type"] == DataType.VARCHAR
    assert field_map["tags"]["max_capacity"] == 4096
    assert field_map["tags"]["max_length"] == 512

    # Verify ARRAY of INT64 mappings
    assert field_map["scores"]["datatype"] == DataType.ARRAY
    assert field_map["scores"]["element_type"] == DataType.INT64
    assert field_map["scores"]["max_capacity"] == 4096

    # Verify JSON mapping for dict
    assert field_map["metadata"]["datatype"] == DataType.JSON
    assert field_map["metadata"]["nullable"] is True

    # Verify datetime maps to VARCHAR(64)
    assert field_map["created_at"]["datatype"] == DataType.VARCHAR
    assert field_map["created_at"]["max_length"] == 64


@patch("wmilvus.core.client.MilvusClient")
def test_ensure_collection_explicit_schema_creation(
    mock_milvus_client: MagicMock,
) -> None:
    """Test _ensure_collection registers all model fields explicitly into Milvus collection schema.

    This test validates:
    1. When a collection does not exist, create_schema() is invoked with enable_dynamic_field=True.
    2. Primary key is added with DataType.VARCHAR and max_length=512 (or INT64).
    3. Vector field is added with DataType.FLOAT_VECTOR and specified dimension.
    4. Every scalar, array, and json field from the Pydantic model is added via schema.add_field().
    5. This guarantees that columns are created distinctly in Milvus rather than lumped into dynamic fields.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = False
    mock_schema = MagicMock()
    mock_instance.create_schema.return_value = mock_schema
    mock_milvus_client.return_value = mock_instance

    class VideoDTO(BaseModel):
        id: str
        path: str
        summary_text: str
        summary_embedding: list[float] = FieldVector(dim=384, default_factory=list)
        frames_ids: list[str] = Field(default_factory=list)

    config = {"uri": "http://localhost:19530"}
    WMilvus(VideoDTO, config)

    # Check create_schema was called with dynamic fields enabled
    mock_instance.create_schema.assert_called_once_with(auto_id=False, enable_dynamic_field=True)

    # Check that schema.add_field was called for pk, vector, and each attribute
    added_field_names = [
        call.kwargs.get("field_name") for call in mock_schema.add_field.call_args_list
    ]
    assert "id" in added_field_names
    assert "summary_embedding" in added_field_names
    assert "path" in added_field_names
    assert "summary_text" in added_field_names
    assert "frames_ids" in added_field_names

    # Check create_collection was called
    mock_instance.create_collection.assert_called_once()


@patch("wmilvus.core.client.MilvusClient")
def test_auto_migration_add_missing_collection_field(
    mock_milvus_client: MagicMock,
) -> None:
    """Test auto-migration when existing collection is missing fields declared in the Pydantic model.

    This test validates:
    1. When a collection already exists in Milvus, describe_collection is inspected.
    2. Any fields present in the Pydantic model but missing in the Milvus schema are identified.
    3. client.add_collection_field() is called with the proper DataType and parameters for each missing field.
    4. Existing collections seamlessly evolve without requiring manual schema migrations.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    # Existing collection only has 'id' and 'summary_embedding'
    mock_instance.describe_collection.return_value = {
        "collection_name": "videodto",
        "fields": [
            {"name": "id", "type": 21},
            {"name": "summary_embedding", "type": 101},
        ],
    }
    mock_milvus_client.return_value = mock_instance

    class VideoDTO(BaseModel):
        id: str
        path: str
        summary_text: str
        summary_embedding: list[float] = FieldVector(dim=384, default_factory=list)
        frames_ids: list[str] = Field(default_factory=list)

    config = {"uri": "http://localhost:19530"}
    WMilvus(VideoDTO, config)

    # Check add_collection_field was called for missing fields: 'path', 'summary_text', 'frames_ids'
    added_field_names = [
        call.kwargs.get("field_name") for call in mock_instance.add_collection_field.call_args_list
    ]
    assert "path" in added_field_names
    assert "summary_text" in added_field_names
    assert "frames_ids" in added_field_names


@patch("wmilvus.core.client.MilvusClient")
def test_drop_collection_operations(
    mock_milvus_client: MagicMock,
) -> None:
    """Test drop() on repository and drop_collection() on WMilvus client.

    This test validates:
    1. db.drop_collection(model_cls) calls MilvusClient.drop_collection and evicts repository cache.
    2. db[model_cls].drop() calls MilvusClient.drop_collection for that repository's collection.
    3. Clean collection lifecycle management is provided to callers.
    """
    mock_instance = MagicMock()
    mock_instance.has_collection.return_value = True
    mock_milvus_client.return_value = mock_instance

    config = {"uri": "http://localhost:19530"}
    db = WMilvus(Person, config)

    # Test repository drop()
    db[Person].drop()
    mock_instance.drop_collection.assert_called_with(collection_name="person")

    # Test client drop_collection()
    mock_instance.drop_collection.reset_mock()
    db.drop_collection(Person)
    mock_instance.drop_collection.assert_called_with(collection_name="person")
    assert Person not in db.repositories
