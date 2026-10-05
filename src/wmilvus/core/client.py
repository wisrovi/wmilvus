"""WMilvus Pydantic ORM Repository and Multi-Collection Client."""

import asyncio
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union
from loguru import logger
import numpy as np
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient

from wmilvus.exceptions import CollectionError, ConnectionError, ValidationError, VectorSearchError
from wmilvus.types import FieldVector, ForensicModel, SearchMatch, VectorRecord

GHOST_AUDIT_LOG_COLLECTION = "_forensic_audit_log"


def get_model_tablename(model_cls: Type[BaseModel]) -> str:
    """Extract table/collection name from Pydantic model class attribute or class name."""
    if hasattr(model_cls, "__tablename__"):
        return getattr(model_cls, "__tablename__")
    if hasattr(model_cls, "__collection_name__"):
        return getattr(model_cls, "__collection_name__")
    return model_cls.__name__.lower()


def inspect_model_vector_field(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Inspect Pydantic model fields to identify the vector field and its configuration."""
    for field_name, field_info in model_cls.model_fields.items():
        json_extra = field_info.json_schema_extra
        if isinstance(json_extra, dict) and json_extra.get("is_vector"):
            return {
                "field_name": field_name,
                "dim": json_extra.get("dim", 128),
                "metric_type": json_extra.get("metric_type", "COSINE"),
                "index_type": json_extra.get("index_type", "HNSW"),
                "params": json_extra.get("params", {"M": 16, "efConstruction": 200}),
            }
    # Fallback default if no explicit FieldVector is annotated
    return {
        "field_name": "vector",
        "dim": 128,
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }


def ensure_ghost_audit_log_collection(client: MilvusClient) -> None:
    """Ensure that the global _forensic_audit_log collection exists in Milvus."""
    if not client.has_collection(collection_name=GHOST_AUDIT_LOG_COLLECTION):
        client.create_collection(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            dimension=2,
            metric_type="L2",
            auto_id=True,
            id_type=DataType.INT64,
        )
        logger.info(f"Created global ghost audit log collection '{GHOST_AUDIT_LOG_COLLECTION}'")


def record_ghost_audit(
    client: MilvusClient,
    action_type: str,
    table_name: str,
    record_id: str,
    data_before: Optional[Dict[str, Any]] = None,
    data_after: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = 1,
) -> None:
    """Record an audit trail event in the global _forensic_audit_log collection."""
    ensure_ghost_audit_log_collection(client)
    now_str = datetime.now(timezone.utc).isoformat()
    payload = {
        "vector": [0.0, 0.0],
        "action_type": action_type,
        "table_name": table_name,
        "record_id": str(record_id),
        "data_before": json.dumps(data_before, default=str) if data_before else "",
        "data_after": json.dumps(data_after, default=str) if data_after else "",
        "create_by": user_id or 1,
        "create_in": now_str,
    }
    client.insert(collection_name=GHOST_AUDIT_LOG_COLLECTION, data=[payload])


class CollectionRepository:
    """Repository bound to a specific Pydantic model class for single-collection ORM operations."""

    def __init__(self, client: MilvusClient, model_cls: Type[BaseModel], forensic: bool = False) -> None:
        self.client = client
        self.model_cls = model_cls
        self.collection_name = get_model_tablename(model_cls)
        self.vector_meta = inspect_model_vector_field(model_cls)
        self.forensic = forensic or issubclass(model_cls, ForensicModel)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_meta["dim"],
                metric_type=self.vector_meta["metric_type"],
                auto_id=False,
                id_type=DataType.VARCHAR,
                max_length=64,
            )
            logger.info(f"Created collection '{self.collection_name}' for model {self.model_cls.__name__}")

    def insert(self, record: BaseModel, user_id: Optional[int] = 1) -> BaseModel:
        """Insert a single typed Pydantic model instance into Milvus with forensic audit logging."""
        data_dict = record.model_dump()
        vector_field_name = self.vector_meta["field_name"]
        if self.forensic and user_id is not None:
            data_dict["create_by"] = user_id
            data_dict["create_in"] = datetime.now(timezone.utc).isoformat()

        rec_id = str(data_dict.get("id", ""))
        payload = {
            "id": rec_id,
            "vector": data_dict[vector_field_name],
            **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
        }
        self.client.upsert(collection_name=self.collection_name, data=[payload])

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="INSERT",
                table_name=self.collection_name,
                record_id=rec_id,
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
            )
        return record

    def insert_batch(
        self,
        records: List[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """Perform bulk insertion of vector records in configurable batch chunks."""
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}

        vector_field_name = self.vector_meta["field_name"]
        total_records = len(records)
        batches_processed = 0

        for i in range(0, total_records, batch_size):
            chunk = records[i : i + batch_size]
            payload_batch = []
            for record in chunk:
                data_dict = record.model_dump()
                if self.forensic and user_id is not None:
                    data_dict["create_by"] = user_id
                    data_dict["create_in"] = datetime.now(timezone.utc).isoformat()
                rec_id = str(data_dict.get("id", ""))
                payload = {
                    "id": rec_id,
                    "vector": data_dict[vector_field_name],
                    **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
                }
                payload_batch.append(payload)

            self.client.upsert(collection_name=self.collection_name, data=payload_batch)
            batches_processed += 1
            logger.info(f"Inserted batch {batches_processed} ({len(chunk)} records) into '{self.collection_name}'")

        if self.forensic and records:
            record_ghost_audit(
                client=self.client,
                action_type="BATCH_INSERT",
                table_name=self.collection_name,
                record_id=f"batch_{total_records}_items",
                data_after={"total_records": total_records, "batches": batches_processed},
                user_id=user_id,
            )

        return {"inserted_count": total_records, "batches_processed": batches_processed}

    def get_all(self, limit: int = 100) -> List[BaseModel]:
        """Query all records from the collection up to the specified limit."""
        raw_items = self.client.query(
            collection_name=self.collection_name,
            filter='id != ""',
            output_fields=["*"],
            limit=limit,
        )
        vector_field_name = self.vector_meta["field_name"]
        results: List[BaseModel] = []
        for item in raw_items:
            obj_data = {k: v for k, v in item.items() if k != "vector"}
            if "vector" in item:
                obj_data[vector_field_name] = item["vector"]
            results.append(self.model_cls.model_validate(obj_data))
        return results

    def get_by_field(self, **kwargs: Any) -> Optional[BaseModel]:
        """Fetch a single record matching scalar field kwargs."""
        if not kwargs:
            return None
        field_name, field_value = next(iter(kwargs.items()))
        filter_expr = f'{field_name} == "{field_value}"' if isinstance(field_value, str) else f"{field_name} == {field_value}"
        raw_items = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["*"],
            limit=1,
        )
        if not raw_items:
            return None
        item = raw_items[0]
        vector_field_name = self.vector_meta["field_name"]
        obj_data = {k: v for k, v in item.items() if k != "vector"}
        if "vector" in item:
            obj_data[vector_field_name] = item["vector"]
        return self.model_cls.model_validate(obj_data)

    def search_similar(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> List[BaseModel]:
        """Perform vector similarity search returning typed Pydantic model instances."""
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        search_output = self.client.search(
            collection_name=self.collection_name,
            data=[vec_list],
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
        )
        vector_field_name = self.vector_meta["field_name"]
        results: List[BaseModel] = []
        if search_output and len(search_output) > 0:
            for item in search_output[0]:
                entity = item.get("entity", {})
                obj_data = {k: v for k, v in entity.items() if k != "vector"}
                if "vector" in entity:
                    obj_data[vector_field_name] = entity["vector"]
                results.append(self.model_cls.model_validate(obj_data))
        return results

    def search_similar_with_scores(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> List[SearchMatch]:
        """Perform vector similarity search returning SearchMatch instances with similarity scores."""
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        search_output = self.client.search(
            collection_name=self.collection_name,
            data=[vec_list],
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
        )
        results: List[SearchMatch] = []
        if search_output and len(search_output) > 0:
            for item in search_output[0]:
                entity = item.get("entity", {})
                distance = float(item.get("distance", 0.0))
                rec_id = str(item.get("id", entity.get("id", "")))
                metadata = {k: v for k, v in entity.items() if k != "vector"}
                results.append(SearchMatch(id=rec_id, distance=distance, metadata=metadata))
        return results

    def search_range(
        self,
        vector: Union[List[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
    ) -> List[BaseModel]:
        """Perform vector similarity range search filtering by minimum similarity score / radius threshold."""
        matches_scores = self.search_range_with_scores(vector=vector, radius=radius, top_k=top_k, filter_expr=filter_expr)
        results: List[BaseModel] = []
        for match in matches_scores:
            rec = self.get_by_field(id=match.id)
            if rec:
                results.append(rec)
        return results

    def search_range_with_scores(
        self,
        vector: Union[List[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
    ) -> List[SearchMatch]:
        """Perform vector similarity range search returning SearchMatch items above radius score."""
        all_matches = self.search_similar_with_scores(vector=vector, top_k=top_k, filter_expr=filter_expr)
        return [m for m in all_matches if m.distance >= radius]

    def search_hybrid(
        self,
        vector: Union[List[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
    ) -> List[BaseModel]:
        """Perform hybrid similarity search combining dense vector search with text/scalar filtering expressions."""
        combined_filter = filter_expr
        if text_query and text_field:
            text_expr = f'{text_field} like "%{text_query}%"'
            combined_filter = f"({filter_expr}) and ({text_expr})" if filter_expr else text_expr

        return self.search_similar(vector=vector, top_k=top_k, filter_expr=combined_filter)

    def verify_schema(self) -> Dict[str, Any]:
        """Verify that the Milvus collection schema matches the bound Pydantic model definition."""
        if not self.client.has_collection(collection_name=self.collection_name):
            raise ValidationError(f"Collection '{self.collection_name}' does not exist in Milvus")

        desc = self.client.describe_collection(collection_name=self.collection_name)
        dim = self.vector_meta["dim"]
        metric_type = self.vector_meta["metric_type"]

        fields = desc.get("fields", [])
        vector_field_found = False
        mismatches = []

        for f in fields:
            if f.get("name") == "vector":
                vector_field_found = True
                params = f.get("params", {})
                coll_dim = params.get("dim") or f.get("dim")
                if coll_dim and int(coll_dim) != int(dim):
                    mismatches.append(f"Dimension mismatch: model expects {dim}, collection has {coll_dim}")

        if not vector_field_found:
            mismatches.append("Vector field 'vector' not found in collection schema")

        if mismatches:
            raise ValidationError(f"Schema verification failed for collection '{self.collection_name}': {'; '.join(mismatches)}")

        return {
            "status": "VALID",
            "collection_name": self.collection_name,
            "dimension": dim,
            "metric_type": metric_type,
            "field_count": len(fields),
        }

    def update(self, record_id: str, record: BaseModel, user_id: Optional[int] = None) -> BaseModel:
        """Update existing record in Milvus collection with forensic audit logging."""
        before_item = self.get_by_field(id=record_id)
        data_before = before_item.model_dump() if before_item else None

        data_dict = record.model_dump()
        vector_field_name = self.vector_meta["field_name"]
        if self.forensic and user_id is not None:
            data_dict["update_by"] = user_id
            data_dict["update_in"] = datetime.now(timezone.utc).isoformat()
        payload = {
            "id": str(record_id),
            "vector": data_dict[vector_field_name],
            **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
        }
        self.client.upsert(collection_name=self.collection_name, data=[payload])

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="UPDATE",
                table_name=self.collection_name,
                record_id=str(record_id),
                data_before={k: v for k, v in data_before.items() if k != vector_field_name} if data_before else None,
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
            )

        return record

    def delete(self, record_id: str, user_id: Optional[int] = None, hard: bool = True) -> None:
        """Delete record by ID from Milvus collection with forensic audit logging."""
        before_item = self.get_by_field(id=record_id)
        data_before = before_item.model_dump() if before_item else None
        vector_field_name = self.vector_meta["field_name"]

        self.client.delete(collection_name=self.collection_name, ids=[str(record_id)])

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="HARD_DELETE" if hard else "SOFT_DELETE",
                table_name=self.collection_name,
                record_id=str(record_id),
                data_before={k: v for k, v in data_before.items() if k != vector_field_name} if data_before else None,
                user_id=user_id,
            )


class WMilvus:
    """High-level Milvus client wrapper designed for simplified Pydantic ORM pipeline usage."""

    def __init__(
        self,
        models: Union[Type[BaseModel], List[Type[BaseModel]]],
        config: Optional[Union[str, Dict[str, Any]]] = None,
        forensic: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, str):
            self.uri = config
            token = kwargs.get("token", "")
            db_name = kwargs.get("db_name", "default")
            timeout = kwargs.get("timeout", 30.0)
        elif isinstance(config, dict):
            self.uri = config.get("uri", "http://localhost:19530")
            token = config.get("token", "")
            db_name = config.get("db_name", "default")
            timeout = config.get("timeout", 30.0)
        else:
            self.uri = kwargs.get("uri", "http://localhost:19530")
            token = kwargs.get("token", "")
            db_name = kwargs.get("db_name", "default")
            timeout = kwargs.get("timeout", 30.0)

        try:
            self.client = MilvusClient(
                uri=self.uri,
                token=token,
                db_name=db_name,
                timeout=timeout,
            )
            logger.info(f"Successfully connected WMilvus ORM to {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus server at '{self.uri}': {e}") from e

        model_list = models if isinstance(models, list) else [models]
        if not model_list:
            raise ValidationError("WMilvus requires at least one Pydantic model class")

        self.repositories: Dict[Type[BaseModel], CollectionRepository] = {}
        self._attr_repos: Dict[str, CollectionRepository] = {}

        for model_cls in model_list:
            repo = CollectionRepository(self.client, model_cls, forensic=forensic)
            self.repositories[model_cls] = repo
            table_name = get_model_tablename(model_cls)
            self._attr_repos[table_name] = repo
            self._attr_repos[model_cls.__name__.lower()] = repo

    def __getitem__(self, item: Type[BaseModel]) -> CollectionRepository:
        if item in self.repositories:
            return self.repositories[item]
        raise CollectionError(f"No repository registered for model '{item.__name__}'")

    def __getattr__(self, name: str) -> CollectionRepository:
        attr_key = name.lower()
        if attr_key in self._attr_repos:
            return self._attr_repos[attr_key]
        raise AttributeError(f"'WMilvus' object has no collection repository named '{name}'")

    def insert(self, record: BaseModel, user_id: Optional[int] = 1) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.insert(record, user_id=user_id)

    def insert_batch(self, records: List[BaseModel], batch_size: int = 1000, user_id: Optional[int] = 1) -> Dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return repo.insert_batch(records, batch_size=batch_size, user_id=user_id)

    def get_all(self, limit: int = 100) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_all(limit=limit)
        raise CollectionError("get_all() on main WMilvus instance is only valid in single-collection mode")

    def get_by_field(self, **kwargs: Any) -> Optional[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_by_field(**kwargs)
        raise CollectionError("get_by_field() on main WMilvus instance is only valid in single-collection mode")

    def search_similar(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_similar(vector=vector, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_similar() on main WMilvus instance is only valid in single-collection mode")

    def search_similar_with_scores(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> List[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_similar_with_scores(vector=vector, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_similar_with_scores() on main WMilvus instance is only valid in single-collection mode")

    def search_range(
        self,
        vector: Union[List[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
    ) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_range(vector=vector, radius=radius, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_range() on main WMilvus instance is only valid in single-collection mode")

    def search_range_with_scores(
        self,
        vector: Union[List[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
    ) -> List[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_range_with_scores(vector=vector, radius=radius, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_range_with_scores() on main WMilvus instance is only valid in single-collection mode")

    def search_hybrid(
        self,
        vector: Union[List[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
    ) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_hybrid(vector=vector, text_query=text_query, text_field=text_field, filter_expr=filter_expr, top_k=top_k)
        raise CollectionError("search_hybrid() on main WMilvus instance is only valid in single-collection mode")

    def verify_schema(self) -> Dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.verify_schema()
        raise CollectionError("verify_schema() on main WMilvus instance is only valid in single-collection mode")

    def update(self, record_id: str, record: BaseModel, user_id: Optional[int] = None) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.update(record_id, record, user_id=user_id)

    def delete(self, record_id: str, user_id: Optional[int] = None) -> None:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            single_repo.delete(record_id, user_id=user_id)
        else:
            raise CollectionError("delete() on main WMilvus instance requires specifying repository in multi-collection mode")

    def get_ghost_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recorded audit events from global _forensic_audit_log ghost collection."""
        raw_items = self.client.query(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            filter='id != 0',
            output_fields=["*"],
            limit=limit,
        )
        return raw_items

    def close(self) -> None:
        if hasattr(self.client, "close"):
            try:
                self.client.close()
                logger.info("Closed WMilvus ORM client connection")
            except Exception as e:
                logger.warning(f"Error closing Milvus client: {e}")

    def __enter__(self) -> "WMilvus":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AsyncCollectionRepository:
    """Async repository bound to a specific Pydantic model for non-blocking ORM operations."""

    def __init__(self, sync_repo: CollectionRepository) -> None:
        self._repo = sync_repo

    async def insert(self, record: BaseModel, user_id: Optional[int] = 1) -> BaseModel:
        return await asyncio.to_thread(self._repo.insert, record, user_id=user_id)

    async def insert_batch(self, records: List[BaseModel], batch_size: int = 1000, user_id: Optional[int] = 1) -> Dict[str, Any]:
        return await asyncio.to_thread(self._repo.insert_batch, records, batch_size=batch_size, user_id=user_id)

    async def get_all(self, limit: int = 100) -> List[BaseModel]:
        return await asyncio.to_thread(self._repo.get_all, limit=limit)

    async def get_by_field(self, **kwargs: Any) -> Optional[BaseModel]:
        return await asyncio.to_thread(self._repo.get_by_field, **kwargs)

    async def search_similar(self, vector: Union[List[float], np.ndarray], top_k: int = 5, filter_expr: str = "") -> List[BaseModel]:
        return await asyncio.to_thread(self._repo.search_similar, vector, top_k=top_k, filter_expr=filter_expr)

    async def search_similar_with_scores(self, vector: Union[List[float], np.ndarray], top_k: int = 5, filter_expr: str = "") -> List[SearchMatch]:
        return await asyncio.to_thread(self._repo.search_similar_with_scores, vector, top_k=top_k, filter_expr=filter_expr)

    async def search_range(self, vector: Union[List[float], np.ndarray], radius: float = 0.8, top_k: int = 10, filter_expr: str = "") -> List[BaseModel]:
        return await asyncio.to_thread(self._repo.search_range, vector, radius=radius, top_k=top_k, filter_expr=filter_expr)

    async def search_range_with_scores(self, vector: Union[List[float], np.ndarray], radius: float = 0.8, top_k: int = 10, filter_expr: str = "") -> List[SearchMatch]:
        return await asyncio.to_thread(self._repo.search_range_with_scores, vector, radius=radius, top_k=top_k, filter_expr=filter_expr)

    async def search_hybrid(self, vector: Union[List[float], np.ndarray], text_query: Optional[str] = None, text_field: Optional[str] = None, filter_expr: str = "", top_k: int = 5) -> List[BaseModel]:
        return await asyncio.to_thread(self._repo.search_hybrid, vector, text_query=text_query, text_field=text_field, filter_expr=filter_expr, top_k=top_k)

    async def verify_schema(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._repo.verify_schema)

    async def update(self, record_id: str, record: BaseModel, user_id: Optional[int] = None) -> BaseModel:
        return await asyncio.to_thread(self._repo.update, record_id, record, user_id=user_id)

    async def delete(self, record_id: str, user_id: Optional[int] = None) -> None:
        return await asyncio.to_thread(self._repo.delete, record_id, user_id=user_id)


class AsyncWMilvus:
    """High-level Async Milvus ORM client for asyncio application usage."""

    def __init__(
        self,
        models: Union[Type[BaseModel], List[Type[BaseModel]]],
        config: Optional[Union[str, Dict[str, Any]]] = None,
        forensic: bool = False,
        **kwargs: Any,
    ) -> None:
        self._sync_client = WMilvus(models, config=config, forensic=forensic, **kwargs)
        self.repositories: Dict[Type[BaseModel], AsyncCollectionRepository] = {
            cls: AsyncCollectionRepository(repo) for cls, repo in self._sync_client.repositories.items()
        }
        self._attr_repos: Dict[str, AsyncCollectionRepository] = {
            k: AsyncCollectionRepository(repo) for k, repo in self._sync_client._attr_repos.items()
        }

    def __getitem__(self, item: Type[BaseModel]) -> AsyncCollectionRepository:
        if item in self.repositories:
            return self.repositories[item]
        raise CollectionError(f"No async repository registered for model '{item.__name__}'")

    def __getattr__(self, name: str) -> AsyncCollectionRepository:
        attr_key = name.lower()
        if attr_key in self._attr_repos:
            return self._attr_repos[attr_key]
        raise AttributeError(f"'AsyncWMilvus' object has no collection repository named '{name}'")

    async def insert(self, record: BaseModel, user_id: Optional[int] = 1) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.insert(record, user_id=user_id)

    async def insert_batch(self, records: List[BaseModel], batch_size: int = 1000, user_id: Optional[int] = 1) -> Dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return await repo.insert_batch(records, batch_size=batch_size, user_id=user_id)

    async def get_all(self, limit: int = 100) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_all(limit=limit)
        raise CollectionError("get_all() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def get_by_field(self, **kwargs: Any) -> Optional[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_by_field(**kwargs)
        raise CollectionError("get_by_field() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def search_similar(self, vector: Union[List[float], np.ndarray], top_k: int = 5, filter_expr: str = "") -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_similar(vector=vector, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_similar() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def search_similar_with_scores(self, vector: Union[List[float], np.ndarray], top_k: int = 5, filter_expr: str = "") -> List[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_similar_with_scores(vector=vector, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_similar_with_scores() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def search_range(self, vector: Union[List[float], np.ndarray], radius: float = 0.8, top_k: int = 10, filter_expr: str = "") -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_range(vector=vector, radius=radius, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_range() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def search_range_with_scores(self, vector: Union[List[float], np.ndarray], radius: float = 0.8, top_k: int = 10, filter_expr: str = "") -> List[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_range_with_scores(vector=vector, radius=radius, top_k=top_k, filter_expr=filter_expr)
        raise CollectionError("search_range_with_scores() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def search_hybrid(self, vector: Union[List[float], np.ndarray], text_query: Optional[str] = None, text_field: Optional[str] = None, filter_expr: str = "", top_k: int = 5) -> List[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_hybrid(vector=vector, text_query=text_query, text_field=text_field, filter_expr=filter_expr, top_k=top_k)
        raise CollectionError("search_hybrid() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def verify_schema(self) -> Dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.verify_schema()
        raise CollectionError("verify_schema() on main AsyncWMilvus instance is only valid in single-collection mode")

    async def update(self, record_id: str, record: BaseModel, user_id: Optional[int] = None) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.update(record_id, record, user_id=user_id)

    async def delete(self, record_id: str, user_id: Optional[int] = None) -> None:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            await single_repo.delete(record_id, user_id=user_id)
        else:
            raise CollectionError("delete() on main AsyncWMilvus instance requires specifying repository in multi-collection mode")

    async def get_ghost_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_client.get_ghost_audit_log, limit=limit)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_client.close)

    async def __aenter__(self) -> "AsyncWMilvus":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
