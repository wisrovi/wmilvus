"""WMilvus Pydantic ORM Repository and Multi-Collection Client."""

import asyncio
import contextlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient
from wmilvus.exceptions import CollectionError, ConnectionError, ValidationError
from wmilvus.types import ForensicModel, SearchMatch

GHOST_AUDIT_LOG_COLLECTION = "_forensic_audit_log"


def get_model_tablename(model_cls: type[BaseModel]) -> str:
    """Extract table/collection name from Pydantic model class attribute or class name."""
    if hasattr(model_cls, "__tablename__"):
        return model_cls.__tablename__
    if hasattr(model_cls, "__collection_name__"):
        return model_cls.__collection_name__
    return model_cls.__name__.lower()


def inspect_model_vector_field(model_cls: type[BaseModel]) -> dict[str, Any]:
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


def inspect_model_pk_field(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Inspect model to identify primary key field name, whether it is integer, and auto_id preference."""
    fields = model_cls.model_fields
    if "id" in fields:
        field_info = fields["id"]
        annotation = field_info.annotation
        import typing

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        is_int = annotation is int or (origin is typing.Union and int in args)
        return {
            "name": "id",
            "is_int": is_int,
            "auto_id": is_int or field_info.default is None,
        }
    return {"name": "id", "is_int": False, "auto_id": False}


def build_filter_expr(filter_kwargs: dict[str, Any]) -> str:
    """Translate dictionary or keyword arguments into Milvus filter expressions."""
    if not filter_kwargs:
        return ""
    clauses = []
    for raw_key, val in filter_kwargs.items():
        if "__" in raw_key:
            field_name, op = raw_key.split("__", 1)
        else:
            field_name, op = raw_key, "eq"

        if op == "eq":
            clause = (
                f'{field_name} == "{val}"' if isinstance(val, str) else f"{field_name} == {val}"
            )
        elif op == "ne":
            clause = (
                f'{field_name} != "{val}"' if isinstance(val, str) else f"{field_name} != {val}"
            )
        elif op == "gt":
            clause = f"{field_name} > {val}"
        elif op == "gte":
            clause = f"{field_name} >= {val}"
        elif op == "lt":
            clause = f"{field_name} < {val}"
        elif op == "lte":
            clause = f"{field_name} <= {val}"
        elif op == "like":
            clause = f'{field_name} like "{val}"'
        elif op == "in" and isinstance(val, (list, tuple, set)):
            formatted_vals = [f'"{v}"' if isinstance(v, str) else str(v) for v in val]
            clause = f"{field_name} in [{', '.join(formatted_vals)}]"
        else:
            clause = (
                f'{field_name} == "{val}"' if isinstance(val, str) else f"{field_name} == {val}"
            )
        clauses.append(clause)

    return " and ".join(clauses)


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
    data_before: Optional[dict[str, Any]] = None,
    data_after: Optional[dict[str, Any]] = None,
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

    def __init__(
        self,
        client: MilvusClient,
        model_cls: type[BaseModel],
        forensic: bool = False,
        consistency_level: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model_cls = model_cls
        self.collection_name = get_model_tablename(model_cls)
        self.vector_meta = inspect_model_vector_field(model_cls)
        self.forensic = forensic or issubclass(model_cls, ForensicModel)
        self.consistency_level = consistency_level
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.has_collection(collection_name=self.collection_name):
            pk_info = inspect_model_pk_field(self.model_cls)
            id_type = DataType.INT64 if pk_info["is_int"] else DataType.VARCHAR
            auto_id = pk_info["auto_id"]
            vector_field_name = self.vector_meta["field_name"]

            try:
                schema = self.client.create_schema(
                    auto_id=auto_id,
                    enable_dynamic_field=True,
                )
                if pk_info["is_int"]:
                    schema.add_field(
                        field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=auto_id
                    )
                else:
                    schema.add_field(
                        field_name="id",
                        datatype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=64,
                        auto_id=auto_id,
                    )

                schema.add_field(
                    field_name=vector_field_name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=self.vector_meta["dim"],
                )

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name=vector_field_name,
                    index_type=self.vector_meta["index_type"],
                    metric_type=self.vector_meta["metric_type"],
                    params=self.vector_meta.get("params", {"M": 16, "efConstruction": 200}),
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info(
                    f"Created collection '{self.collection_name}' with explicit schema for model {self.model_cls.__name__}"
                )
            except Exception as e:
                logger.debug(f"Explicit schema creation fallback: {e}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.vector_meta["dim"],
                    metric_type=self.vector_meta["metric_type"],
                    auto_id=auto_id,
                    id_type=id_type,
                    max_length=64,
                )
                logger.info(
                    f"Created collection '{self.collection_name}' for model {self.model_cls.__name__}"
                )
        else:
            try:
                load_state = self.client.get_load_state(collection_name=self.collection_name)
                if isinstance(load_state, dict):
                    state_val = str(load_state.get("state", ""))
                    if "NotLoad" in state_val or "NotExist" in state_val:
                        self.client.load_collection(collection_name=self.collection_name)
                        logger.info(
                            f"Loaded existing collection '{self.collection_name}' into memory"
                        )
            except Exception as e:
                logger.debug(f"Could not verify or load collection '{self.collection_name}': {e}")

    def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        """Insert a single typed Pydantic model instance into Milvus with forensic audit logging."""
        data_dict = record.model_dump(mode="json")
        vector_field_name = self.vector_meta["field_name"]
        if self.forensic and user_id is not None:
            data_dict["create_by"] = user_id
            data_dict["create_in"] = datetime.now(timezone.utc).isoformat()

        rec_id = data_dict.get("id")
        has_id = rec_id is not None and rec_id != ""

        payload = {
            vector_field_name: data_dict[vector_field_name],
            **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
        }
        if has_id:
            payload["id"] = rec_id

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        if has_id:
            self.client.upsert(collection_name=self.collection_name, data=[payload], **kwargs)
        else:
            insert_res = self.client.insert(
                collection_name=self.collection_name, data=[payload], **kwargs
            )
            if isinstance(insert_res, dict) and insert_res.get("ids"):
                generated_id = insert_res["ids"][0]
                if hasattr(record, "id"):
                    record.id = generated_id
                rec_id = generated_id

        with contextlib.suppress(Exception):
            self.client.flush(collection_name=self.collection_name)

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="INSERT",
                table_name=self.collection_name,
                record_id=str(rec_id or ""),
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
            )
        return record

    def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Perform bulk insertion of vector records in configurable batch chunks."""
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}

        vector_field_name = self.vector_meta["field_name"]
        total_records = len(records)
        batches_processed = 0

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        for i in range(0, total_records, batch_size):
            chunk = records[i : i + batch_size]
            payload_batch = []
            has_ids = True

            for record in chunk:
                data_dict = record.model_dump(mode="json")
                if self.forensic and user_id is not None:
                    data_dict["create_by"] = user_id
                    data_dict["create_in"] = datetime.now(timezone.utc).isoformat()
                rec_id = data_dict.get("id")
                rec_has_id = rec_id is not None and rec_id != ""
                if not rec_has_id:
                    has_ids = False

                payload = {
                    vector_field_name: data_dict[vector_field_name],
                    **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
                }
                if rec_has_id:
                    payload["id"] = rec_id
                payload_batch.append(payload)

            if has_ids:
                self.client.upsert(
                    collection_name=self.collection_name, data=payload_batch, **kwargs
                )
            else:
                res = self.client.insert(
                    collection_name=self.collection_name, data=payload_batch, **kwargs
                )
                if isinstance(res, dict) and res.get("ids"):
                    for rec, gid in zip(chunk, res["ids"]):
                        if hasattr(rec, "id"):
                            rec.id = gid

            batches_processed += 1
            logger.info(
                f"Inserted batch {batches_processed} ({len(chunk)} records) into '{self.collection_name}'"
            )

        with contextlib.suppress(Exception):
            self.client.flush(collection_name=self.collection_name)

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

    def get_all(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        """Query all records from the collection up to the specified limit."""
        kwargs: dict[str, Any] = {}
        if partition_names:
            kwargs["partition_names"] = partition_names
        if self.consistency_level:
            kwargs.setdefault("consistency_level", self.consistency_level)

        vector_field_name = self.vector_meta["field_name"]
        if output_fields is None:
            actual_output_fields = (
                ["*", vector_field_name]
                if include_vector and vector_field_name != "vector"
                else (["*", "vector"] if include_vector else ["*"])
            )
        else:
            actual_output_fields = output_fields

        raw_items = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=actual_output_fields,
            limit=limit,
            **kwargs,
        )
        results: list[BaseModel] = []
        for item in raw_items:
            obj_data = {k: v for k, v in item.items() if k != "vector"}
            if "vector" in item:
                obj_data[vector_field_name] = item["vector"]
            results.append(self.model_cls.model_validate(obj_data))
        return results

    def get_by_field(
        self,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        """Fetch a single record matching scalar field kwargs."""
        if not kwargs:
            return None
        field_name, field_value = next(iter(kwargs.items()))
        filter_expr = (
            f'{field_name} == "{field_value}"'
            if isinstance(field_value, str)
            else f"{field_name} == {field_value}"
        )
        extra_kwargs: dict[str, Any] = {}
        if partition_names:
            extra_kwargs["partition_names"] = partition_names
        if self.consistency_level:
            extra_kwargs.setdefault("consistency_level", self.consistency_level)

        vector_field_name = self.vector_meta["field_name"]
        if output_fields is None:
            actual_output_fields = (
                ["*", vector_field_name]
                if include_vector and vector_field_name != "vector"
                else (["*", "vector"] if include_vector else ["*"])
            )
        else:
            actual_output_fields = output_fields

        raw_items = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=actual_output_fields,
            limit=1,
            **extra_kwargs,
        )
        if not raw_items:
            return None
        item = raw_items[0]
        vector_field_name = self.vector_meta["field_name"]
        obj_data = {k: v for k, v in item.items() if k != "vector"}
        if "vector" in item:
            obj_data[vector_field_name] = item["vector"]
        return self.model_cls.model_validate(obj_data)

    def filter_by(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        """Query records using typed keyword arguments (e.g. name='Alice', age__gte=18)."""
        filter_expr = build_filter_expr(kwargs)
        extra_kwargs: dict[str, Any] = {}
        if partition_names:
            extra_kwargs["partition_names"] = partition_names
        if self.consistency_level:
            extra_kwargs.setdefault("consistency_level", self.consistency_level)

        vector_field_name = self.vector_meta["field_name"]
        if output_fields is None:
            actual_output_fields = (
                ["*", vector_field_name]
                if include_vector and vector_field_name != "vector"
                else (["*", "vector"] if include_vector else ["*"])
            )
        else:
            actual_output_fields = output_fields

        raw_items = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=actual_output_fields,
            limit=limit,
            **extra_kwargs,
        )
        vector_field_name = self.vector_meta["field_name"]
        results: list[BaseModel] = []
        for item in raw_items:
            obj_data = {k: v for k, v in item.items() if k != "vector"}
            if "vector" in item:
                obj_data[vector_field_name] = item["vector"]
            results.append(self.model_cls.model_validate(obj_data))
        return results

    def iter_all(
        self,
        batch_size: int = 1000,
        filter_expr: str = "",
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> Iterator[list[BaseModel]]:
        """Yield batches of typed Pydantic models using native Milvus query iterator."""
        vector_field_name = self.vector_meta["field_name"]
        kwargs: dict[str, Any] = {}
        if partition_names:
            kwargs["partition_names"] = partition_names

        if output_fields is None:
            actual_output_fields = (
                ["*", vector_field_name]
                if include_vector and vector_field_name != "vector"
                else (["*", "vector"] if include_vector else ["*"])
            )
        else:
            actual_output_fields = output_fields

        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=batch_size,
            filter=filter_expr,
            output_fields=actual_output_fields,
            **kwargs,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                results = []
                for item in batch:
                    obj_data = {k: v for k, v in item.items() if k != "vector"}
                    if "vector" in item:
                        obj_data[vector_field_name] = item["vector"]
                    results.append(self.model_cls.model_validate(obj_data))
                yield results
        finally:
            iterator.close()

    def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        """Perform vector similarity search returning typed Pydantic model instances."""
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        vector_field_name = self.vector_meta["field_name"]
        kwargs: dict[str, Any] = {}
        if search_params:
            kwargs["search_params"] = search_params
        if partition_names:
            kwargs["partition_names"] = partition_names
        if self.consistency_level:
            kwargs.setdefault("consistency_level", self.consistency_level)

        if output_fields is None:
            actual_output_fields = (
                ["*", vector_field_name]
                if include_vector and vector_field_name != "vector"
                else (["*", "vector"] if include_vector else ["*"])
            )
        else:
            actual_output_fields = output_fields

        search_output = self.client.search(
            collection_name=self.collection_name,
            data=[vec_list],
            limit=top_k,
            filter=filter_expr,
            output_fields=actual_output_fields,
            anns_field=vector_field_name,
            **kwargs,
        )
        results: list[BaseModel] = []
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
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        """Perform vector similarity search returning SearchMatch instances with similarity scores."""
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        vector_field_name = self.vector_meta["field_name"]
        kwargs: dict[str, Any] = {}
        if search_params:
            kwargs["search_params"] = search_params
        if partition_names:
            kwargs["partition_names"] = partition_names
        if self.consistency_level:
            kwargs.setdefault("consistency_level", self.consistency_level)

        search_output = self.client.search(
            collection_name=self.collection_name,
            data=[vec_list],
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
            anns_field=vector_field_name,
            **kwargs,
        )
        results: list[SearchMatch] = []
        if search_output and len(search_output) > 0:
            for item in search_output[0]:
                entity = item.get("entity", {})
                distance = float(item.get("distance", 0.0))
                rec_id = str(item.get("id", entity.get("id", "")))
                metadata = {
                    k: v for k, v in entity.items() if k not in ("vector", vector_field_name)
                }
                results.append(SearchMatch(id=rec_id, distance=distance, metadata=metadata))
        return results

    def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        """Perform vector similarity range search filtering by minimum similarity score / radius threshold."""
        matches_scores = self.search_range_with_scores(
            vector=vector,
            radius=radius,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
        )
        results: list[BaseModel] = []
        for match in matches_scores:
            rec = self.get_by_field(id=match.id, partition_names=partition_names)
            if rec:
                results.append(rec)
        return results

    def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        """Perform vector similarity range search returning SearchMatch items above radius score."""
        all_matches = self.search_similar_with_scores(
            vector=vector,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
        )
        return [m for m in all_matches if m.distance >= radius]

    def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        """Perform hybrid similarity search combining dense vector search with text/scalar filtering expressions."""
        combined_filter = filter_expr
        if text_query and text_field:
            text_expr = f'{text_field} like "%{text_query}%"'
            combined_filter = f"({filter_expr}) and ({text_expr})" if filter_expr else text_expr

        return self.search_similar(
            vector=vector,
            top_k=top_k,
            filter_expr=combined_filter,
            search_params=search_params,
            partition_names=partition_names,
            output_fields=output_fields,
            include_vector=include_vector,
        )

    def verify_schema(self) -> dict[str, Any]:
        """Verify that the Milvus collection schema matches the bound Pydantic model definition."""
        if not self.client.has_collection(collection_name=self.collection_name):
            raise ValidationError(f"Collection '{self.collection_name}' does not exist in Milvus")

        desc = self.client.describe_collection(collection_name=self.collection_name)
        dim = self.vector_meta["dim"]
        metric_type = self.vector_meta["metric_type"]

        fields = desc.get("fields", [])
        vector_field_found = False
        mismatches = []

        vector_field_name = self.vector_meta["field_name"]
        for f in fields:
            if f.get("name") in (vector_field_name, "vector"):
                vector_field_found = True
                params = f.get("params", {})
                coll_dim = params.get("dim") or f.get("dim")
                if coll_dim and int(coll_dim) != int(dim):
                    mismatches.append(
                        f"Dimension mismatch: model expects {dim}, collection has {coll_dim}"
                    )

        if not vector_field_found:
            mismatches.append(f"Vector field '{vector_field_name}' not found in collection schema")

        if mismatches:
            raise ValidationError(
                f"Schema verification failed for collection '{self.collection_name}': {'; '.join(mismatches)}"
            )

        return {
            "status": "VALID",
            "collection_name": self.collection_name,
            "dimension": dim,
            "metric_type": metric_type,
            "field_count": len(fields),
        }

    def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        """Update existing record in Milvus collection with forensic audit logging."""
        before_item = self.get_by_field(
            id=record_id, partition_names=[partition_name] if partition_name else None
        )
        data_before = before_item.model_dump(mode="json") if before_item else None

        data_dict = record.model_dump(mode="json")
        vector_field_name = self.vector_meta["field_name"]

        # Preserve vector if incoming record has empty/missing vector but existing record has one
        incoming_vec = data_dict.get(vector_field_name)
        if (
            incoming_vec is None or (isinstance(incoming_vec, list) and not incoming_vec)
        ) and before_item:
            existing_vec = getattr(before_item, vector_field_name, None)
            if existing_vec:
                data_dict[vector_field_name] = (
                    existing_vec.tolist()
                    if isinstance(existing_vec, np.ndarray)
                    else list(existing_vec)
                )
                with contextlib.suppress(Exception):
                    setattr(record, vector_field_name, data_dict[vector_field_name])

        if self.forensic and user_id is not None:
            data_dict["update_by"] = user_id
            data_dict["update_in"] = datetime.now(timezone.utc).isoformat()
        payload = {
            "id": record_id,
            vector_field_name: data_dict[vector_field_name],
            **{k: v for k, v in data_dict.items() if k not in ("id", vector_field_name)},
        }
        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        self.client.upsert(collection_name=self.collection_name, data=[payload], **kwargs)

        with contextlib.suppress(Exception):
            self.client.flush(collection_name=self.collection_name)

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="UPDATE",
                table_name=self.collection_name,
                record_id=str(record_id),
                data_before={k: v for k, v in data_before.items() if k != vector_field_name}
                if data_before
                else None,
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
            )

        return record

    def delete(
        self,
        record_id: Union[str, int],
        user_id: Optional[int] = None,
        hard: bool = True,
        partition_name: Optional[str] = None,
    ) -> None:
        """Delete record by ID from Milvus collection with forensic audit logging."""
        before_item = self.get_by_field(
            id=record_id, partition_names=[partition_name] if partition_name else None
        )
        data_before = before_item.model_dump(mode="json") if before_item else None
        vector_field_name = self.vector_meta["field_name"]

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        self.client.delete(collection_name=self.collection_name, ids=[record_id], **kwargs)

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="HARD_DELETE" if hard else "SOFT_DELETE",
                table_name=self.collection_name,
                record_id=str(record_id),
                data_before={k: v for k, v in data_before.items() if k != vector_field_name}
                if data_before
                else None,
                user_id=user_id,
            )

    def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        user_id: Optional[int] = None,
        hard: bool = True,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete a batch of records by IDs from Milvus collection."""
        if not record_ids:
            return {"delete_count": 0}

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        res = self.client.delete(collection_name=self.collection_name, ids=record_ids, **kwargs)

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="BATCH_DELETE",
                table_name=self.collection_name,
                record_id=f"batch_{len(record_ids)}_ids",
                data_before={"ids": [str(x) for x in record_ids]},
                user_id=user_id,
            )
        return res

    def delete_by_expr(
        self,
        filter_expr: str,
        user_id: Optional[int] = None,
        hard: bool = True,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete records matching a filter expression from Milvus collection."""
        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        res = self.client.delete(collection_name=self.collection_name, filter=filter_expr, **kwargs)

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="EXPR_DELETE",
                table_name=self.collection_name,
                record_id=filter_expr[:64],
                data_before={"filter": filter_expr},
                user_id=user_id,
            )
        return res


class WMilvus:
    """High-level Milvus client wrapper designed for simplified Pydantic ORM pipeline usage."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        config = kwargs.pop("config", None)
        forensic = kwargs.pop("forensic", False)
        models = None
        uri = None

        if args:
            if isinstance(args[0], str):
                uri = args[0]
                models = args[1] if len(args) > 1 else kwargs.pop("models", None)
            else:
                models = args[0]
                if len(args) > 1:
                    config = args[1]
        else:
            models = kwargs.pop("models", None)

        if not uri:
            if isinstance(config, str):
                uri = config
            elif isinstance(config, dict):
                uri = config.get("uri", "http://localhost:19530")
            else:
                uri = kwargs.get("uri", "http://localhost:19530")

        self.uri = uri
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

        self.repositories: dict[type[BaseModel], CollectionRepository] = {}
        self._attr_repos: dict[str, CollectionRepository] = {}

        consistency_level = kwargs.get("consistency_level")
        for model_cls in model_list:
            repo = CollectionRepository(
                self.client,
                model_cls,
                forensic=forensic,
                consistency_level=consistency_level,
            )
            self.repositories[model_cls] = repo
            table_name = get_model_tablename(model_cls)
            self._attr_repos[table_name] = repo
            self._attr_repos[model_cls.__name__.lower()] = repo

    def __getitem__(self, item: type[BaseModel]) -> CollectionRepository:
        if item in self.repositories:
            return self.repositories[item]
        raise CollectionError(f"No repository registered for model '{item.__name__}'")

    def __getattr__(self, name: str) -> CollectionRepository:
        attr_key = name.lower()
        if attr_key in self._attr_repos:
            return self._attr_repos[attr_key]
        raise AttributeError(f"'WMilvus' object has no collection repository named '{name}'")

    def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.insert(record, user_id=user_id, partition_name=partition_name)

    def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return repo.insert_batch(
            records, batch_size=batch_size, user_id=user_id, partition_name=partition_name
        )

    def get_all(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "get_all() on main WMilvus instance is only valid in single-collection mode"
        )

    def get_by_field(
        self,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        raise CollectionError(
            "get_by_field() on main WMilvus instance is only valid in single-collection mode"
        )

    def filter_by(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        raise CollectionError(
            "filter_by() on main WMilvus instance is only valid in single-collection mode"
        )

    def iter_all(
        self,
        batch_size: int = 1000,
        filter_expr: str = "",
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> Iterator[list[BaseModel]]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            yield from single_repo.iter_all(
                batch_size=batch_size,
                filter_expr=filter_expr,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        else:
            raise CollectionError(
                "iter_all() on main WMilvus instance is only valid in single-collection mode"
            )

    def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_similar(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "search_similar() on main WMilvus instance is only valid in single-collection mode"
        )

    def search_similar_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_similar_with_scores() on main WMilvus instance is only valid in single-collection mode"
        )

    def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_range(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_range() on main WMilvus instance is only valid in single-collection mode"
        )

    def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_range_with_scores(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_range_with_scores() on main WMilvus instance is only valid in single-collection mode"
        )

    def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_hybrid(
                vector=vector,
                text_query=text_query,
                text_field=text_field,
                filter_expr=filter_expr,
                top_k=top_k,
                search_params=search_params,
                partition_names=partition_names,
                ranker=ranker,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "search_hybrid() on main WMilvus instance is only valid in single-collection mode"
        )

    def verify_schema(self) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.verify_schema()
        raise CollectionError(
            "verify_schema() on main WMilvus instance is only valid in single-collection mode"
        )

    def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.update(record_id, record, user_id=user_id, partition_name=partition_name)

    def delete(
        self,
        record_id: Union[str, int],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            single_repo.delete(record_id, user_id=user_id, partition_name=partition_name)
        else:
            raise CollectionError(
                "delete() on main WMilvus instance requires specifying repository in multi-collection mode"
            )

    def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        raise CollectionError(
            "delete_batch() on main WMilvus instance requires specifying repository in multi-collection mode"
        )

    def delete_by_expr(
        self,
        filter_expr: str,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        raise CollectionError(
            "delete_by_expr() on main WMilvus instance requires specifying repository in multi-collection mode"
        )

    def get_ghost_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch recorded audit events from global _forensic_audit_log ghost collection."""
        raw_items = self.client.query(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            filter="",
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

    async def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        return await asyncio.to_thread(
            self._repo.insert, record, user_id=user_id, partition_name=partition_name
        )

    async def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._repo.insert_batch,
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
        )

    async def get_all(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        return await asyncio.to_thread(
            self._repo.get_all,
            limit=limit,
            partition_names=partition_names,
            output_fields=output_fields,
            include_vector=include_vector,
        )

    async def get_by_field(
        self,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        return await asyncio.to_thread(
            self._repo.get_by_field,
            partition_names=partition_names,
            output_fields=output_fields,
            include_vector=include_vector,
            **kwargs,
        )

    async def filter_by(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        return await asyncio.to_thread(
            self._repo.filter_by,
            limit=limit,
            partition_names=partition_names,
            output_fields=output_fields,
            include_vector=include_vector,
            **kwargs,
        )

    async def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        return await asyncio.to_thread(
            self._repo.search_similar,
            vector,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
            output_fields=output_fields,
            include_vector=include_vector,
        )

    async def search_similar_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        return await asyncio.to_thread(
            self._repo.search_similar_with_scores,
            vector,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
        )

    async def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        return await asyncio.to_thread(
            self._repo.search_range,
            vector,
            radius=radius,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
        )

    async def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        return await asyncio.to_thread(
            self._repo.search_range_with_scores,
            vector,
            radius=radius,
            top_k=top_k,
            filter_expr=filter_expr,
            search_params=search_params,
            partition_names=partition_names,
        )

    async def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        return await asyncio.to_thread(
            self._repo.search_hybrid,
            vector,
            text_query=text_query,
            text_field=text_field,
            filter_expr=filter_expr,
            top_k=top_k,
            search_params=search_params,
            partition_names=partition_names,
            ranker=ranker,
            output_fields=output_fields,
            include_vector=include_vector,
        )

    async def verify_schema(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._repo.verify_schema)

    async def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        return await asyncio.to_thread(
            self._repo.update,
            record_id,
            record,
            user_id=user_id,
            partition_name=partition_name,
        )

    async def delete(
        self,
        record_id: Union[str, int],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        return await asyncio.to_thread(
            self._repo.delete, record_id, user_id=user_id, partition_name=partition_name
        )

    async def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._repo.delete_batch,
            record_ids,
            user_id=user_id,
            partition_name=partition_name,
        )

    async def delete_by_expr(
        self,
        filter_expr: str,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._repo.delete_by_expr,
            filter_expr,
            user_id=user_id,
            partition_name=partition_name,
        )


class AsyncWMilvus:
    """High-level Async Milvus ORM client for asyncio application usage."""

    def __init__(
        self,
        models: Union[type[BaseModel], list[type[BaseModel]]],
        config: Optional[Union[str, dict[str, Any]]] = None,
        forensic: bool = False,
        **kwargs: Any,
    ) -> None:
        self._sync_client = WMilvus(models, config=config, forensic=forensic, **kwargs)
        self.repositories: dict[type[BaseModel], AsyncCollectionRepository] = {
            cls: AsyncCollectionRepository(repo)
            for cls, repo in self._sync_client.repositories.items()
        }
        self._attr_repos: dict[str, AsyncCollectionRepository] = {
            k: AsyncCollectionRepository(repo) for k, repo in self._sync_client._attr_repos.items()
        }

    def __getitem__(self, item: type[BaseModel]) -> AsyncCollectionRepository:
        if item in self.repositories:
            return self.repositories[item]
        raise CollectionError(f"No async repository registered for model '{item.__name__}'")

    def __getattr__(self, name: str) -> AsyncCollectionRepository:
        attr_key = name.lower()
        if attr_key in self._attr_repos:
            return self._attr_repos[attr_key]
        raise AttributeError(f"'AsyncWMilvus' object has no collection repository named '{name}'")

    async def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.insert(record, user_id=user_id, partition_name=partition_name)

    async def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return await repo.insert_batch(
            records, batch_size=batch_size, user_id=user_id, partition_name=partition_name
        )

    async def get_all(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "get_all() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def get_by_field(
        self,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        raise CollectionError(
            "get_by_field() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def filter_by(
        self,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        raise CollectionError(
            "filter_by() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_similar(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "search_similar() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def search_similar_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_similar_with_scores() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_range(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_range() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_range_with_scores(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        raise CollectionError(
            "search_range_with_scores() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_hybrid(
                vector=vector,
                text_query=text_query,
                text_field=text_field,
                filter_expr=filter_expr,
                top_k=top_k,
                search_params=search_params,
                partition_names=partition_names,
                ranker=ranker,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        raise CollectionError(
            "search_hybrid() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def verify_schema(self) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.verify_schema()
        raise CollectionError(
            "verify_schema() on main AsyncWMilvus instance is only valid in single-collection mode"
        )

    async def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.update(record_id, record, user_id=user_id, partition_name=partition_name)

    async def delete(
        self,
        record_id: Union[str, int],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            await single_repo.delete(record_id, user_id=user_id, partition_name=partition_name)
        else:
            raise CollectionError(
                "delete() on main AsyncWMilvus instance requires specifying repository in multi-collection mode"
            )

    async def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        raise CollectionError(
            "delete_batch() on main AsyncWMilvus instance requires specifying repository in multi-collection mode"
        )

    async def delete_by_expr(
        self,
        filter_expr: str,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        raise CollectionError(
            "delete_by_expr() on main AsyncWMilvus instance requires specifying repository in multi-collection mode"
        )

    async def get_ghost_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync_client.get_ghost_audit_log, limit=limit)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_client.close)

    async def __aenter__(self) -> "AsyncWMilvus":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
