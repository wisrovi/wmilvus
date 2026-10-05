"""WMilvus Pydantic ORM Repository and Multi-Collection Client."""

import inspect
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union
from loguru import logger
import numpy as np
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient

from wmilvus.exceptions import CollectionError, ConnectionError, VectorSearchError
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

    for field_name in ("vector", "embedding", "vec", "face_vec", "image_vec", "frame_vec"):
        if field_name in model_cls.model_fields:
            return {
                "field_name": field_name,
                "dim": 128,
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }

    return {
        "field_name": "vector",
        "dim": 128,
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }


def ensure_ghost_audit_log_collection(client: MilvusClient) -> None:
    """Ensure global _forensic_audit_log collection exists in Milvus."""
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
        """Insert a single Pydantic model instance into Milvus collection."""
        data_dict = record.model_dump()
        vector_field_name = self.vector_meta["field_name"]

        if vector_field_name not in data_dict or data_dict[vector_field_name] is None:
            raise CollectionError(f"Record missing required vector field '{vector_field_name}'")

        if self.forensic:
            now = datetime.now(timezone.utc)
            if "create_by" in data_dict and data_dict["create_by"] is None:
                data_dict["create_by"] = user_id
            if "create_in" in data_dict and data_dict["create_in"] is None:
                data_dict["create_in"] = now.isoformat()

        payload = {
            "id": str(data_dict.get("id", "")),
            "vector": data_dict[vector_field_name],
            **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
        }

        self.client.upsert(collection_name=self.collection_name, data=[payload])

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="INSERT",
                table_name=self.collection_name,
                record_id=str(data_dict.get("id", "")),
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
            )

        return record

    def get_all(self, limit: int = 100) -> List[BaseModel]:
        """Fetch all records from collection mapped back to Pydantic models."""
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
    """High-level Milvus ORM and Multi-Collection Client using Pydantic models."""

    def __init__(
        self,
        target: Optional[
            Union[
                Type[BaseModel],
                List[Type[BaseModel]],
                Dict[str, Any],
                str,
            ]
        ] = None,
        db_config: Optional[Dict[str, Any]] = None,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
        timeout: Optional[float] = 30.0,
        forensic: bool = False,
    ) -> None:
        resolved_config = db_config or {}
        if isinstance(target, dict):
            resolved_config = target
            target_models: List[Type[BaseModel]] = []
        elif isinstance(target, list):
            target_models = target
        elif inspect.isclass(target) and issubclass(target, BaseModel):
            target_models = [target]
        else:
            target_models = []

        conn_uri = resolved_config.get("uri", uri)
        conn_token = resolved_config.get("token", token)
        conn_dbname = resolved_config.get("db_name", db_name)
        conn_timeout = resolved_config.get("timeout", timeout)

        self.uri = conn_uri
        try:
            self.client = MilvusClient(
                uri=conn_uri,
                token=conn_token,
                db_name=conn_dbname,
                timeout=conn_timeout,
            )
            logger.info(f"Successfully connected WMilvus ORM to {conn_uri}")
        except Exception as e:
            logger.error(f"Failed to connect WMilvus ORM to {conn_uri}: {e}")
            raise ConnectionError(f"Connection failed to {conn_uri}: {e}") from e

        self.repositories: Dict[Any, CollectionRepository] = {}
        self._attr_repos: Dict[str, CollectionRepository] = {}
        self.forensic = forensic

        for model_cls in target_models:
            repo = CollectionRepository(self.client, model_cls, forensic=forensic)
            self.repositories[model_cls] = repo
            table_name = get_model_tablename(model_cls)
            self._attr_repos[table_name.lower()] = repo
            self._attr_repos[model_cls.__name__.lower()] = repo

    def get_ghost_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recorded entries from the global _forensic_audit_log collection."""
        ensure_ghost_audit_log_collection(self.client)
        return self.client.query(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            filter='table_name != ""',
            output_fields=["*"],
            limit=limit,
        )

    def __getitem__(self, item: Type[BaseModel]) -> CollectionRepository:
        if item in self.repositories:
            return self.repositories[item]
        if inspect.isclass(item) and issubclass(item, BaseModel):
            repo = CollectionRepository(self.client, item, forensic=self.forensic)
            self.repositories[item] = repo
            table_name = get_model_tablename(item)
            self._attr_repos[table_name.lower()] = repo
            self._attr_repos[item.__name__.lower()] = repo
            return repo
        raise KeyError(f"No collection repository registered for model {item}")

    def __getattr__(self, name: str) -> CollectionRepository:
        attr_key = name.lower()
        if attr_key in self._attr_repos:
            return self._attr_repos[attr_key]
        raise AttributeError(f"'WMilvus' object has no collection repository named '{name}'")

    def insert(self, record: BaseModel, user_id: Optional[int] = 1) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.insert(record, user_id=user_id)

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
