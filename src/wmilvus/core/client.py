"""WMilvus Pydantic ORM Repository and Multi-Collection Client."""

import asyncio
import atexit
import concurrent.futures
import contextlib
import json
import os
import queue
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient
from wmilvus.exceptions import CollectionError, ConnectionError, ValidationError, VectorSearchError
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
                "has_vector": True,
            }
    # Fallback if a field is named 'vector' in model fields
    if "vector" in model_cls.model_fields:
        return {
            "field_name": "vector",
            "dim": 128,
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
            "has_vector": True,
        }
    # Pure scalar / non-vector model: provision dummy vector metadata for Milvus compatibility
    return {
        "field_name": "_dummy_vector",
        "dim": 8,
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {},
        "has_vector": False,
    }


def inspect_model_pk_field(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Inspect model to identify primary key field name, whether it is integer, and auto_id preference."""
    fields = model_cls.model_fields
    for candidate in ("id", "id_"):
        if candidate in fields:
            field_info = fields[candidate]
            annotation = field_info.annotation
            import typing

            origin = typing.get_origin(annotation)
            args = typing.get_args(annotation)
            is_int = annotation is int or (origin is typing.Union and int in args)
            return {
                "name": candidate,
                "is_int": is_int,
                "auto_id": is_int or field_info.default is None,
            }
    return {"name": "id", "is_int": False, "auto_id": False}


def inspect_model_fields_schema(
    model_cls: type[BaseModel], pk_name: str, vector_field_name: str
) -> list[dict[str, Any]]:
    """Inspect Pydantic model fields to generate explicit Milvus collection schema definitions.

    Translates Pydantic types into appropriate Milvus DataType entries (VARCHAR, INT64,
    DOUBLE, BOOL, ARRAY, JSON) so that every field has its own dedicated column in Milvus,
    preventing fields from being collapsed into dynamic fields ($meta).
    """
    import typing
    from datetime import date, datetime

    schema_fields: list[dict[str, Any]] = []

    for field_name, field_info in model_cls.model_fields.items():
        if field_name in (pk_name, vector_field_name):
            continue

        annotation = field_info.annotation

        # Unwrap Optional / Union[..., None]
        origin = typing.get_origin(annotation)
        if origin is typing.Union:
            args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                annotation = args[0]
                origin = typing.get_origin(annotation)

        # 1. Strings
        if annotation is str:
            max_len = 65535
            if field_info.metadata:
                for meta in field_info.metadata:
                    if hasattr(meta, "max_length") and meta.max_length:
                        max_len = min(int(meta.max_length), 65535)
                        break
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.VARCHAR,
                    "max_length": max_len,
                    "nullable": True,
                }
            )

        # 2. Integers
        elif annotation is int:
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.INT64,
                    "nullable": True,
                }
            )

        # 3. Floats
        elif annotation is float:
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.DOUBLE,
                    "nullable": True,
                }
            )

        # 4. Booleans
        elif annotation is bool:
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.BOOL,
                    "nullable": True,
                }
            )

        # 5. Dates and datetimes (serialized as ISO strings in json mode)
        elif annotation in (datetime, date):
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.VARCHAR,
                    "max_length": 64,
                    "nullable": True,
                }
            )

        # 6. Dictionaries / Mappings -> DataType.JSON
        elif origin in (dict, typing.Mapping, dict) or annotation is dict:
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.JSON,
                    "nullable": True,
                }
            )

        # 7. Lists / Sequences
        elif origin in (list, typing.Sequence, list) or annotation is list:
            item_args = typing.get_args(annotation)
            item_type = item_args[0] if item_args else None
            if item_type and typing.get_origin(item_type) is typing.Union:
                union_args = [a for a in typing.get_args(item_type) if a is not type(None)]
                if len(union_args) == 1:
                    item_type = union_args[0]

            if item_type is str:
                schema_fields.append(
                    {
                        "field_name": field_name,
                        "datatype": DataType.ARRAY,
                        "element_type": DataType.VARCHAR,
                        "max_capacity": 4096,
                        "max_length": 512,
                        "nullable": True,
                    }
                )
            elif item_type is int:
                schema_fields.append(
                    {
                        "field_name": field_name,
                        "datatype": DataType.ARRAY,
                        "element_type": DataType.INT64,
                        "max_capacity": 4096,
                        "nullable": True,
                    }
                )
            elif item_type is float:
                schema_fields.append(
                    {
                        "field_name": field_name,
                        "datatype": DataType.ARRAY,
                        "element_type": DataType.DOUBLE,
                        "max_capacity": 4096,
                        "nullable": True,
                    }
                )
            elif item_type is bool:
                schema_fields.append(
                    {
                        "field_name": field_name,
                        "datatype": DataType.ARRAY,
                        "element_type": DataType.BOOL,
                        "max_capacity": 4096,
                        "nullable": True,
                    }
                )
            else:
                schema_fields.append(
                    {
                        "field_name": field_name,
                        "datatype": DataType.JSON,
                        "nullable": True,
                    }
                )

        # 8. Arbitrary JSON or sub-models
        elif annotation is typing.Any or (
            isinstance(annotation, type) and issubclass(annotation, BaseModel)
        ):
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.JSON,
                    "nullable": True,
                }
            )

        # 9. Fallback string
        else:
            schema_fields.append(
                {
                    "field_name": field_name,
                    "datatype": DataType.VARCHAR,
                    "max_length": 65535,
                    "nullable": True,
                }
            )

    return schema_fields


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
        elif op == "contains":
            clause = (
                f'array_contains({field_name}, "{val}")'
                if isinstance(val, str)
                else f"array_contains({field_name}, {val})"
            )
        elif op == "in" and isinstance(val, (list, tuple, set)):
            formatted_vals = [f'"{v}"' if isinstance(v, str) else str(v) for v in val]
            clause = f"{field_name} in [{', '.join(formatted_vals)}]"
        else:
            clause = (
                f'{field_name} == "{val}"' if isinstance(val, str) else f"{field_name} == {val}"
            )
        clauses.append(clause)

    return " and ".join(clauses)


_VERIFIED_GHOST_COLLECTIONS: set[int] = set()
_GLOBAL_CLEANUP_REGISTRY: list[Any] = []
_CLEANUP_LOCK = threading.Lock()


def _cleanup_all_active_buffers() -> None:
    """Flush and shut down all active WriteBuffer and ForensicAuditDispatcher instances upon interpreter exit."""
    with _CLEANUP_LOCK:
        for item in list(_GLOBAL_CLEANUP_REGISTRY):
            with contextlib.suppress(Exception):
                item.close()
        _GLOBAL_CLEANUP_REGISTRY.clear()


atexit.register(_cleanup_all_active_buffers)


def ensure_ghost_audit_log_collection(client: MilvusClient) -> None:
    """Ensure that the global _forensic_audit_log collection exists in Milvus (cached in memory)."""
    client_verified = getattr(client, "_wmilvus_verified_collections", None)
    if isinstance(client_verified, set) and GHOST_AUDIT_LOG_COLLECTION in client_verified:
        return

    client_id = id(client)
    if client_id in _VERIFIED_GHOST_COLLECTIONS:
        return

    if not client.has_collection(collection_name=GHOST_AUDIT_LOG_COLLECTION):
        client.create_collection(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            dimension=2,
            metric_type="L2",
            auto_id=True,
            id_type=DataType.INT64,
        )
        logger.info(f"Created global ghost audit log collection '{GHOST_AUDIT_LOG_COLLECTION}'")

    _VERIFIED_GHOST_COLLECTIONS.add(client_id)
    if not isinstance(client_verified, set):
        client_verified = set()
        with contextlib.suppress(Exception):
            client._wmilvus_verified_collections = client_verified
    client_verified.add(GHOST_AUDIT_LOG_COLLECTION)


class ForensicAuditDispatcher:
    """Background asynchronous dispatcher for forensic audit logging in Milvus."""

    def __init__(
        self,
        client: MilvusClient,
        batch_size: int = 50,
        flush_interval: float = 0.025,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=50000)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        with _CLEANUP_LOCK:
            _GLOBAL_CLEANUP_REGISTRY.append(self)

    def record(self, payload: dict[str, Any]) -> None:
        """Enqueue an audit event to be dispatched asynchronously."""
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._flush_batch([payload])

    def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        ensure_ghost_audit_log_collection(self.client)
        try:
            self.client.insert(collection_name=GHOST_AUDIT_LOG_COLLECTION, data=batch)
        except Exception as e:
            logger.error(f"Failed to persist forensic audit batch ({len(batch)} records): {e}")

    def _worker_loop(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.time()
        while not self._stop_event.is_set():
            try:
                timeout = max(0.005, self.flush_interval - (time.time() - last_flush))
                item = self._queue.get(timeout=timeout)
                if item is None:
                    self._queue.task_done()
                    break
                batch.append(item)
                if (
                    len(batch) >= self.batch_size
                    or (time.time() - last_flush) >= self.flush_interval
                ):
                    self._flush_batch(batch)
                    for _ in range(len(batch)):
                        self._queue.task_done()
                    batch = []
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    for _ in range(len(batch)):
                        self._queue.task_done()
                    batch = []
                    last_flush = time.time()

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    batch.append(item)
                else:
                    self._queue.task_done()
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)
            for _ in range(len(batch)):
                self._queue.task_done()

    def flush(self) -> None:
        """Synchronously wait until all queued audit events are persisted."""
        self._queue.join()

    def close(self) -> None:
        """Stop worker and flush remaining audit records."""
        self.flush()
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        with _CLEANUP_LOCK:
            if self in _GLOBAL_CLEANUP_REGISTRY:
                _GLOBAL_CLEANUP_REGISTRY.remove(self)


class WriteBuffer:
    """Thread-safe write-behind micro-batching buffer for high-throughput individual writes."""

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        window_ms: float = 25.0,
        max_size: int = 50,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.window_s = max(0.005, window_ms / 1000.0)
        self.max_size = max_size
        self._queue: queue.Queue[
            tuple[str, dict[str, Any], Optional[concurrent.futures.Future]]
        ] = queue.Queue(maxsize=50000)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        with _CLEANUP_LOCK:
            _GLOBAL_CLEANUP_REGISTRY.append(self)

    def put_upsert(
        self, payload: dict[str, Any], future: Optional[concurrent.futures.Future] = None
    ) -> None:
        self._queue.put(("upsert", payload, future))

    def put_insert(
        self, payload: dict[str, Any], future: Optional[concurrent.futures.Future] = None
    ) -> None:
        self._queue.put(("insert", payload, future))

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            items: list[tuple[str, dict[str, Any], Optional[concurrent.futures.Future]]] = []
            try:
                item = self._queue.get(timeout=self.window_s)
                items.append(item)
                while len(items) < self.max_size:
                    try:
                        it = self._queue.get_nowait()
                        items.append(it)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if items:
                self._flush_items(items)
                for _ in range(len(items)):
                    self._queue.task_done()

        drain_items = []
        while not self._queue.empty():
            try:
                it = self._queue.get_nowait()
                drain_items.append(it)
            except queue.Empty:
                break
        if drain_items:
            self._flush_items(drain_items)
            for _ in range(len(drain_items)):
                self._queue.task_done()

    def _flush_items(
        self,
        items: list[tuple[str, dict[str, Any], Optional[concurrent.futures.Future]]],
    ) -> None:
        upsert_payloads: list[dict[str, Any]] = []
        upsert_futures: list[Optional[concurrent.futures.Future]] = []
        insert_payloads: list[dict[str, Any]] = []
        insert_futures: list[Optional[concurrent.futures.Future]] = []

        for op, payload, fut in items:
            if op == "upsert":
                upsert_payloads.append(payload)
                upsert_futures.append(fut)
            else:
                insert_payloads.append(payload)
                insert_futures.append(fut)

        if upsert_payloads:
            try:
                res = self.client.upsert(collection_name=self.collection_name, data=upsert_payloads)
                for fut in upsert_futures:
                    if fut and not fut.done():
                        fut.set_result(res)
            except Exception as e:
                logger.error(
                    f"WriteBuffer upsert failed for '{self.collection_name}' ({len(upsert_payloads)} records): {e}"
                )
                for fut in upsert_futures:
                    if fut and not fut.done():
                        fut.set_exception(e)

        if insert_payloads:
            try:
                res = self.client.insert(collection_name=self.collection_name, data=insert_payloads)
                ids = res.get("ids", []) if isinstance(res, dict) else []
                for i, fut in enumerate(insert_futures):
                    if fut and not fut.done():
                        gen_id = ids[i] if i < len(ids) else None
                        fut.set_result(gen_id)
            except Exception as e:
                logger.error(
                    f"WriteBuffer insert failed for '{self.collection_name}' ({len(insert_payloads)} records): {e}"
                )
                for fut in insert_futures:
                    if fut and not fut.done():
                        fut.set_exception(e)

    def flush(self) -> None:
        """Wait until all queued operations are flushed to Milvus."""
        self._queue.join()

    def close(self) -> None:
        """Stop worker and flush remaining operations."""
        self.flush()
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        with _CLEANUP_LOCK:
            if self in _GLOBAL_CLEANUP_REGISTRY:
                _GLOBAL_CLEANUP_REGISTRY.remove(self)


def record_ghost_audit(
    client: MilvusClient,
    action_type: str,
    table_name: str,
    record_id: str,
    data_before: Optional[dict[str, Any]] = None,
    data_after: Optional[dict[str, Any]] = None,
    user_id: Optional[int] = 1,
    dispatcher: Optional[ForensicAuditDispatcher] = None,
) -> None:
    """Record an audit trail event in the global _forensic_audit_log collection."""
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
    if dispatcher is not None and isinstance(dispatcher, ForensicAuditDispatcher):
        dispatcher.record(payload)
    else:
        client_disp = getattr(client, "_wmilvus_audit_dispatcher", None)
        if isinstance(client_disp, ForensicAuditDispatcher):
            client_disp.record(payload)
        else:
            ensure_ghost_audit_log_collection(client)
            client.insert(collection_name=GHOST_AUDIT_LOG_COLLECTION, data=[payload])


class CollectionRepository:
    """Repository bound to a specific Pydantic model class for single-collection ORM operations."""

    def __init__(
        self,
        client: MilvusClient,
        model_cls: type[BaseModel],
        forensic: bool = False,
        consistency_level: Optional[str] = None,
        write_buffer: bool = False,
        buffer_window_ms: float = 25.0,
        buffer_max_size: int = 50,
        async_audit: bool = True,
        **kwargs: Any,
    ) -> None:
        self.client = client
        self.model_cls = model_cls
        self.collection_name = get_model_tablename(model_cls)
        self.vector_meta = inspect_model_vector_field(model_cls)
        self.has_vector = bool(self.vector_meta.get("has_vector", True))
        self.pk_info = inspect_model_pk_field(model_cls)
        self.forensic = forensic or issubclass(model_cls, ForensicModel)
        self.consistency_level = consistency_level
        self._ensure_collection()

        self.audit_dispatcher: Optional[ForensicAuditDispatcher] = None
        if self.forensic and async_audit:
            shared_disp = getattr(client, "_wmilvus_audit_dispatcher", None)
            if shared_disp is None and hasattr(client, "insert"):
                try:
                    from unittest.mock import MagicMock

                    is_mock = isinstance(client, MagicMock) or hasattr(client, "_mock_return_value")
                except ImportError:
                    is_mock = False
                if not is_mock or kwargs.get("_force_async_audit", False):
                    shared_disp = ForensicAuditDispatcher(client)
                    client._wmilvus_audit_dispatcher = shared_disp
            self.audit_dispatcher = shared_disp

        self.write_buffer: Optional[WriteBuffer] = None
        if write_buffer:
            try:
                from unittest.mock import MagicMock

                is_mock = isinstance(client, MagicMock) or hasattr(client, "_mock_return_value")
            except ImportError:
                is_mock = False
            if not is_mock or kwargs.get("_force_buffer", False):
                self.write_buffer = WriteBuffer(
                    client=self.client,
                    collection_name=self.collection_name,
                    window_ms=buffer_window_ms,
                    max_size=buffer_max_size,
                )

    def _deserialize_record(self, item: dict[str, Any]) -> BaseModel:
        """Deserialize raw Milvus query dictionary into the bound Pydantic model."""
        vector_field_name = self.vector_meta["field_name"]
        obj_data = {
            k: v for k, v in item.items() if k not in ("vector", "_dummy_vector", vector_field_name)
        }
        if self.has_vector:
            if vector_field_name in item:
                obj_data[vector_field_name] = item[vector_field_name]
            elif "vector" in item:
                obj_data[vector_field_name] = item["vector"]
        return self.model_cls.model_validate(obj_data)

    def _ensure_collection(self) -> None:
        pk_info = self.pk_info
        pk_name = pk_info["name"]
        id_type = DataType.INT64 if pk_info["is_int"] else DataType.VARCHAR
        auto_id = pk_info["auto_id"]
        vector_field_name = self.vector_meta["field_name"]
        scalar_fields = inspect_model_fields_schema(
            self.model_cls, pk_name=pk_name, vector_field_name=vector_field_name
        )

        if not self.client.has_collection(collection_name=self.collection_name):
            try:
                schema = self.client.create_schema(
                    auto_id=auto_id,
                    enable_dynamic_field=True,
                )
                if pk_info["is_int"]:
                    schema.add_field(
                        field_name=pk_name,
                        datatype=DataType.INT64,
                        is_primary=True,
                        auto_id=auto_id,
                    )
                else:
                    schema.add_field(
                        field_name=pk_name,
                        datatype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=512,
                        auto_id=auto_id,
                    )

                schema.add_field(
                    field_name=vector_field_name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=self.vector_meta["dim"],
                )

                for f_info in scalar_fields:
                    schema.add_field(**f_info)

                index_params = self.client.prepare_index_params()
                index_kwargs: dict[str, Any] = {
                    "field_name": vector_field_name,
                    "index_type": self.vector_meta["index_type"],
                    "metric_type": self.vector_meta["metric_type"],
                }
                if self.vector_meta.get("params"):
                    index_kwargs["params"] = self.vector_meta["params"]
                index_params.add_index(**index_kwargs)

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
                    max_length=512,
                )
                if not self.has_vector:
                    self.vector_meta["field_name"] = "vector"
                logger.info(
                    f"Created collection '{self.collection_name}' for model {self.model_cls.__name__}"
                )
        else:
            try:
                desc = self.client.describe_collection(collection_name=self.collection_name)
                if isinstance(desc, dict):
                    existing_fields_list = desc.get("fields", [])
                    if self.has_vector:
                        vector_field_name = self.vector_meta["field_name"]
                        expected_dim = int(self.vector_meta["dim"])
                        for f in existing_fields_list:
                            if isinstance(f, dict) and f.get("name") in (
                                vector_field_name,
                                "vector",
                            ):
                                coll_dim = f.get("params", {}).get("dim") or f.get("dim")
                                if coll_dim and int(coll_dim) != expected_dim:
                                    logger.warning(
                                        f"Dimension mismatch in Milvus collection '{self.collection_name}': "
                                        f"Existing collection has dim={coll_dim}, but model '{self.model_cls.__name__}' specifies dim={expected_dim}. "
                                        f"Milvus vector dimensions are immutable. To use dim={expected_dim}, drop the collection (e.g. db.drop_collection({self.model_cls.__name__})) and recreate it."
                                    )
                    else:
                        for f in existing_fields_list:
                            if isinstance(f, dict) and (
                                f.get("type") in (100, 101)
                                or "vector" in str(f.get("type_name", "")).lower()
                                or f.get("name") in ("_dummy_vector", "vector")
                            ):
                                self.vector_meta["field_name"] = f.get("name")
                                coll_dim = f.get("params", {}).get("dim") or f.get("dim")
                                if coll_dim:
                                    self.vector_meta["dim"] = int(coll_dim)
                                break

                    if hasattr(self.client, "add_collection_field"):
                        existing_fields = {
                            f.get("name")
                            for f in desc.get("fields", [])
                            if isinstance(f, dict) and f.get("name")
                        }
                    for f_info in scalar_fields:
                        f_name = f_info["field_name"]
                        if f_name not in existing_fields:
                            try:
                                field_kwargs = {
                                    k: v
                                    for k, v in f_info.items()
                                    if k not in ("field_name", "datatype")
                                }
                                self.client.add_collection_field(
                                    collection_name=self.collection_name,
                                    field_name=f_name,
                                    data_type=f_info["datatype"],
                                    **field_kwargs,
                                )
                                logger.info(
                                    f"Added missing column '{f_name}' to existing collection '{self.collection_name}'"
                                )
                            except Exception as add_err:
                                logger.debug(
                                    f"Could not auto-add missing column '{f_name}' to '{self.collection_name}': {add_err}"
                                )
            except Exception as schema_err:
                logger.debug(
                    f"Could not inspect/update existing schema for '{self.collection_name}': {schema_err}"
                )

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
        auto_flush: bool = False,
    ) -> BaseModel:
        """Insert a single typed Pydantic model instance into Milvus with forensic audit logging."""
        data_dict = record.model_dump(mode="json")
        vector_field_name = self.vector_meta["field_name"]
        pk_name = self.pk_info["name"]

        if self.forensic and user_id is not None:
            data_dict["create_by"] = user_id
            data_dict["create_in"] = datetime.now(timezone.utc).isoformat()

        rec_id = data_dict.get(pk_name)
        has_id = rec_id is not None and rec_id != ""

        vector_val = (
            data_dict.get(vector_field_name)
            if self.has_vector
            else ([0.0] * self.vector_meta["dim"])
        )
        if vector_val is None:
            vector_val = [0.0] * self.vector_meta["dim"]

        payload = {
            vector_field_name: vector_val,
            **{k: v for k, v in data_dict.items() if k not in (pk_name, vector_field_name)},
        }
        if has_id:
            payload[pk_name] = rec_id

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        if has_id:
            if self.write_buffer:
                self.write_buffer.put_upsert(payload)
            else:
                self.client.upsert(collection_name=self.collection_name, data=[payload], **kwargs)
        else:
            if self.write_buffer and not auto_flush:
                fut = concurrent.futures.Future() if hasattr(record, pk_name) else None
                self.write_buffer.put_insert(payload, fut)
                if fut:
                    with contextlib.suppress(Exception):
                        gen_id = fut.result(timeout=10.0)
                        if gen_id is not None:
                            setattr(record, pk_name, gen_id)
                            rec_id = gen_id
            else:
                insert_res = self.client.insert(
                    collection_name=self.collection_name, data=[payload], **kwargs
                )
                if isinstance(insert_res, dict) and insert_res.get("ids"):
                    generated_id = insert_res["ids"][0]
                    if hasattr(record, pk_name):
                        setattr(record, pk_name, generated_id)
                    rec_id = generated_id

        if auto_flush:
            self.flush()

        if self.forensic:
            record_ghost_audit(
                client=self.client,
                action_type="INSERT",
                table_name=self.collection_name,
                record_id=str(rec_id or ""),
                data_after={k: v for k, v in data_dict.items() if k != vector_field_name},
                user_id=user_id,
                dispatcher=self.audit_dispatcher,
            )
        return record

    def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        """Perform bulk insertion of vector records in configurable batch chunks."""
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}

        vector_field_name = self.vector_meta["field_name"]
        pk_name = self.pk_info["name"]
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
                rec_id = data_dict.get(pk_name)
                rec_has_id = rec_id is not None and rec_id != ""
                if not rec_has_id:
                    has_ids = False

                vector_val = (
                    data_dict.get(vector_field_name)
                    if self.has_vector
                    else ([0.0] * self.vector_meta["dim"])
                )
                if vector_val is None:
                    vector_val = [0.0] * self.vector_meta["dim"]

                payload = {
                    vector_field_name: vector_val,
                    **{k: v for k, v in data_dict.items() if k not in (pk_name, vector_field_name)},
                }
                if rec_has_id:
                    payload[pk_name] = rec_id
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
                        if hasattr(rec, pk_name):
                            setattr(rec, pk_name, gid)

            batches_processed += 1
            logger.info(
                f"Inserted batch {batches_processed} ({len(chunk)} records) into '{self.collection_name}'"
            )

        if auto_flush:
            self.flush()

        if self.forensic and records:
            record_ghost_audit(
                client=self.client,
                action_type="BATCH_INSERT",
                table_name=self.collection_name,
                record_id=f"batch_{total_records}_items",
                data_after={"total_records": total_records, "batches": batches_processed},
                user_id=user_id,
                dispatcher=self.audit_dispatcher,
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
        if not self.has_vector:
            actual_output_fields = output_fields if output_fields is not None else ["*"]
        elif output_fields is None:
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
        return [self._deserialize_record(item) for item in raw_items]

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
        filter_expr = build_filter_expr(kwargs)
        extra_kwargs: dict[str, Any] = {}
        if partition_names:
            extra_kwargs["partition_names"] = partition_names
        if self.consistency_level:
            extra_kwargs.setdefault("consistency_level", self.consistency_level)

        vector_field_name = self.vector_meta["field_name"]
        if not self.has_vector:
            actual_output_fields = output_fields if output_fields is not None else ["*"]
        elif output_fields is None:
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
        return self._deserialize_record(raw_items[0])

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
        if not self.has_vector:
            actual_output_fields = output_fields if output_fields is not None else ["*"]
        elif output_fields is None:
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
        return [self._deserialize_record(item) for item in raw_items]

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

        if not self.has_vector:
            actual_output_fields = output_fields if output_fields is not None else ["*"]
        elif output_fields is None:
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
                yield [self._deserialize_record(item) for item in batch]
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
        if not self.has_vector:
            raise VectorSearchError(
                f"Model '{self.model_cls.__name__}' does not have a vector field. "
                "Vector similarity search is not supported on scalar-only models."
            )
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
                results.append(self._deserialize_record(entity))
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
        if not self.has_vector:
            raise VectorSearchError(
                f"Model '{self.model_cls.__name__}' does not have a vector field. "
                "Vector similarity search is not supported on scalar-only models."
            )
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
                rec_id = str(item.get("id", entity.get(self.pk_info["name"], entity.get("id", ""))))
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
        if not self.has_vector:
            raise VectorSearchError(
                f"Model '{self.model_cls.__name__}' does not have a vector field. "
                "Vector range search is not supported on scalar-only models."
            )
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
        if not self.has_vector:
            raise VectorSearchError(
                f"Model '{self.model_cls.__name__}' does not have a vector field. "
                "Vector range search is not supported on scalar-only models."
            )
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
        if not self.has_vector:
            raise VectorSearchError(
                f"Model '{self.model_cls.__name__}' does not have a vector field. "
                "Hybrid search is not supported on scalar-only models."
            )
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
        fields = desc.get("fields", [])

        if not self.has_vector:
            return {
                "status": "VALID",
                "collection_name": self.collection_name,
                "has_vector": False,
                "fields_count": len(fields),
            }

        dim = self.vector_meta["dim"]
        metric_type = self.vector_meta["metric_type"]

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
        auto_flush: bool = False,
    ) -> BaseModel:
        """Update existing record in Milvus collection with forensic audit logging."""
        pk_name = self.pk_info["name"]
        data_dict = record.model_dump(mode="json")
        vector_field_name = self.vector_meta["field_name"]

        incoming_vec = data_dict.get(vector_field_name) if self.has_vector else None
        has_incoming_vec = bool(
            incoming_vec is not None
            and (not isinstance(incoming_vec, list) or len(incoming_vec) > 0)
        )

        data_before = None
        needs_before = self.forensic or (not has_incoming_vec and self.has_vector)
        if needs_before:
            output_fields = None if not has_incoming_vec else ["*"]
            before_item = self.get_by_field(
                partition_names=[partition_name] if partition_name else None,
                output_fields=output_fields,
                include_vector=not has_incoming_vec,
                **{pk_name: record_id},
            )
            if before_item:
                data_before = before_item.model_dump(mode="json")
                if not has_incoming_vec and self.has_vector:
                    existing_vec = getattr(before_item, vector_field_name, None)
                    if existing_vec:
                        incoming_vec = (
                            existing_vec.tolist()
                            if isinstance(existing_vec, np.ndarray)
                            else list(existing_vec)
                        )
                        with contextlib.suppress(Exception):
                            setattr(record, vector_field_name, incoming_vec)

        if not self.has_vector:
            vector_val = [0.0] * self.vector_meta["dim"]
        else:
            vector_val = incoming_vec or ([0.0] * self.vector_meta["dim"])

        if self.forensic and user_id is not None:
            data_dict["update_by"] = user_id
            data_dict["update_in"] = datetime.now(timezone.utc).isoformat()
        payload = {
            pk_name: record_id,
            vector_field_name: vector_val,
            **{k: v for k, v in data_dict.items() if k not in (pk_name, vector_field_name)},
        }
        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        if self.write_buffer:
            self.write_buffer.put_upsert(payload)
        else:
            self.client.upsert(collection_name=self.collection_name, data=[payload], **kwargs)

        if auto_flush:
            self.flush()

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
                dispatcher=self.audit_dispatcher,
            )

        return record

    def update_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        """Perform bulk update/upsert of records in configurable batch chunks with audit logging."""
        if not records:
            return {"updated_count": 0, "batches_processed": 0}

        vector_field_name = self.vector_meta["field_name"]
        pk_name = self.pk_info["name"]
        total_records = len(records)
        batches_processed = 0

        kwargs: dict[str, Any] = {}
        if partition_name:
            kwargs["partition_name"] = partition_name

        for i in range(0, total_records, batch_size):
            chunk = records[i : i + batch_size]
            payload_batch = []

            for record in chunk:
                data_dict = record.model_dump(mode="json")
                if self.forensic and user_id is not None:
                    data_dict["update_by"] = user_id
                    data_dict["update_in"] = datetime.now(timezone.utc).isoformat()
                rec_id = data_dict.get(pk_name)

                vector_val = (
                    data_dict.get(vector_field_name)
                    if self.has_vector
                    else ([0.0] * self.vector_meta["dim"])
                )
                if vector_val is None:
                    vector_val = [0.0] * self.vector_meta["dim"]

                payload = {
                    pk_name: rec_id,
                    vector_field_name: vector_val,
                    **{k: v for k, v in data_dict.items() if k not in (pk_name, vector_field_name)},
                }
                payload_batch.append(payload)

            self.client.upsert(collection_name=self.collection_name, data=payload_batch, **kwargs)
            batches_processed += 1

        if auto_flush:
            self.flush()

        if self.forensic and records:
            record_ghost_audit(
                client=self.client,
                action_type="BATCH_UPDATE",
                table_name=self.collection_name,
                record_id=f"batch_{total_records}_items",
                data_after={"total_records": total_records, "batches": batches_processed},
                user_id=user_id,
                dispatcher=self.audit_dispatcher,
            )

        return {"updated_count": total_records, "batches_processed": batches_processed}

    def flush(self) -> None:
        """Explicitly flush write buffers, audit dispatchers, and collection data to storage."""
        if self.write_buffer:
            self.write_buffer.flush()
        if self.audit_dispatcher:
            self.audit_dispatcher.flush()
        with contextlib.suppress(Exception):
            self.client.flush(collection_name=self.collection_name)

    def close(self) -> None:
        """Close write buffers and audit dispatchers for this collection repository."""
        if self.write_buffer:
            self.write_buffer.close()
            self.write_buffer = None
        if self.audit_dispatcher:
            self.audit_dispatcher.close()
            self.audit_dispatcher = None

    def delete(
        self,
        record_id: Union[str, int],
        user_id: Optional[int] = None,
        hard: bool = True,
        partition_name: Optional[str] = None,
    ) -> None:
        """Delete record by ID from Milvus collection with forensic audit logging."""
        pk_name = self.pk_info["name"]
        before_item = self.get_by_field(
            partition_names=[partition_name] if partition_name else None,
            **{pk_name: record_id},
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

    def drop(self) -> None:
        """Drop this collection completely from the Milvus database."""
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}' from Milvus")

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

        write_buffer = kwargs.pop("write_buffer", None)
        if write_buffer is None:
            if isinstance(config, dict):
                write_buffer = config.get("write_buffer", False)
            else:
                write_buffer = os.environ.get("WMILVUS_WRITE_BUFFER", "false").lower() in (
                    "true",
                    "1",
                    "yes",
                )

        buffer_window_ms = kwargs.pop("buffer_window_ms", None)
        if buffer_window_ms is None and isinstance(config, dict):
            buffer_window_ms = config.get("buffer_window_ms", 25.0)
        buffer_window_ms = float(buffer_window_ms or 25.0)

        buffer_max_size = kwargs.pop("buffer_max_size", None)
        if buffer_max_size is None and isinstance(config, dict):
            buffer_max_size = config.get("buffer_max_size", 50)
        buffer_max_size = int(buffer_max_size or 50)

        async_audit = kwargs.pop("async_audit", None)
        if async_audit is None and isinstance(config, dict):
            async_audit = config.get("async_audit", True)
        async_audit = True if async_audit is None else bool(async_audit)

        consistency_level = kwargs.get("consistency_level")
        for model_cls in model_list:
            repo = CollectionRepository(
                self.client,
                model_cls,
                forensic=forensic,
                consistency_level=consistency_level,
                write_buffer=bool(write_buffer),
                buffer_window_ms=buffer_window_ms,
                buffer_max_size=buffer_max_size,
                async_audit=async_audit,
                **kwargs,
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

    def _resolve_repo(self, model: Union[type[BaseModel], str]) -> CollectionRepository:
        """Resolve a collection repository by model class or collection name."""
        if isinstance(model, type) and issubclass(model, BaseModel):
            if model in self.repositories:
                return self.repositories[model]
        elif isinstance(model, str):
            key = model.lower()
            if key in self._attr_repos:
                return self._attr_repos[key]
        raise CollectionError(f"No repository registered for model '{model}'")

    def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.insert(
            record, user_id=user_id, partition_name=partition_name, auto_flush=auto_flush
        )

    def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return repo.insert_batch(
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    def get_all(
        self,
        model: Optional[Union[type[BaseModel], str, int]] = None,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if isinstance(model, int):
            limit = model
            model = None
        if model is not None:
            return self._resolve_repo(model).get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        combined: list[BaseModel] = []
        for repo in self.repositories.values():
            combined.extend(
                repo.get_all(
                    limit=limit,
                    partition_names=partition_names,
                    output_fields=output_fields,
                    include_vector=include_vector,
                )
            )
        return combined

    def get_by_field(
        self,
        model: Optional[Union[type[BaseModel], str]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        if model is not None and (
            (isinstance(model, type) and issubclass(model, BaseModel))
            or (isinstance(model, str) and model.lower() in self._attr_repos)
        ):
            return self._resolve_repo(model).get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        if model is not None:
            kwargs["model"] = model
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        for repo in self.repositories.values():
            item = repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
            if item is not None:
                return item
        return None

    def filter_by(
        self,
        model: Optional[Union[type[BaseModel], str]] = None,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        if model is not None and (
            (isinstance(model, type) and issubclass(model, BaseModel))
            or (isinstance(model, str) and model.lower() in self._attr_repos)
        ):
            return self._resolve_repo(model).filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        if model is not None:
            kwargs["model"] = model
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        combined: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                combined.extend(
                    repo.filter_by(
                        limit=limit,
                        partition_names=partition_names,
                        output_fields=output_fields,
                        include_vector=include_vector,
                        **kwargs,
                    )
                )
        return combined

    def iter_all(
        self,
        model: Optional[Union[type[BaseModel], str]] = None,
        batch_size: int = 1000,
        filter_expr: str = "",
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> Iterator[list[BaseModel]]:
        if model is not None:
            yield from self._resolve_repo(model).iter_all(
                batch_size=batch_size,
                filter_expr=filter_expr,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
            return
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            yield from single_repo.iter_all(
                batch_size=batch_size,
                filter_expr=filter_expr,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
            return
        for repo in self.repositories.values():
            yield from repo.iter_all(
                batch_size=batch_size,
                filter_expr=filter_expr,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )

    def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if model is not None:
            return self._resolve_repo(model).search_similar(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                results.extend(
                    repo.search_similar(
                        vector=vector,
                        top_k=top_k,
                        filter_expr=filter_expr,
                        search_params=search_params,
                        partition_names=partition_names,
                        output_fields=output_fields,
                        include_vector=include_vector,
                    )
                )
        return results[:top_k]

    def search_similar_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if model is not None:
            return self._resolve_repo(model).search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        matches: list[SearchMatch] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                matches.extend(
                    repo.search_similar_with_scores(
                        vector=vector,
                        top_k=top_k,
                        filter_expr=filter_expr,
                        search_params=search_params,
                        partition_names=partition_names,
                    )
                )
        matches.sort(key=lambda m: m.distance, reverse=True)
        return matches[:top_k]

    def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        if model is not None:
            return self._resolve_repo(model).search_range(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                results.extend(
                    repo.search_range(
                        vector=vector,
                        radius=radius,
                        top_k=top_k,
                        filter_expr=filter_expr,
                        search_params=search_params,
                        partition_names=partition_names,
                    )
                )
        return results[:top_k]

    def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if model is not None:
            return self._resolve_repo(model).search_range_with_scores(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
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
        matches: list[SearchMatch] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                matches.extend(
                    repo.search_range_with_scores(
                        vector=vector,
                        radius=radius,
                        top_k=top_k,
                        filter_expr=filter_expr,
                        search_params=search_params,
                        partition_names=partition_names,
                    )
                )
        matches.sort(key=lambda m: m.distance, reverse=True)
        return matches[:top_k]

    def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if model is not None:
            return self._resolve_repo(model).search_hybrid(
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                results.extend(
                    repo.search_hybrid(
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
                )
        return results[:top_k]

    def verify_schema(self, model: Optional[Union[type[BaseModel], str]] = None) -> dict[str, Any]:
        if model is not None:
            return self._resolve_repo(model).verify_schema()
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.verify_schema()
        results = {}
        for name_cls, repo in self.repositories.items():
            results[name_cls.__name__] = repo.verify_schema()
        return {"status": "VALID", "collections": results}

    def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return repo.update(
            record_id,
            record,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    def update_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        """Perform bulk update/upsert of records in configurable batch chunks across collections."""
        if not records:
            return {"updated_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return repo.update_batch(
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    def flush(self, model: Optional[Union[type[BaseModel], str]] = None) -> None:
        """Explicitly flush collection data to storage (use sparingly; Milvus handles flushing automatically)."""
        if model is not None:
            self._resolve_repo(model).flush()
        elif len(self.repositories) == 1:
            next(iter(self.repositories.values())).flush()
        else:
            for repo in self.repositories.values():
                repo.flush()

    def delete(
        self,
        record_id: Union[str, int],
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        if model is not None:
            self._resolve_repo(model).delete(
                record_id, user_id=user_id, partition_name=partition_name
            )
            return
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            single_repo.delete(record_id, user_id=user_id, partition_name=partition_name)
            return
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                repo.delete(record_id, user_id=user_id, partition_name=partition_name)

    def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if model is not None:
            return self._resolve_repo(model).delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        total_deleted = 0
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                res = repo.delete_batch(record_ids, user_id=user_id, partition_name=partition_name)
                total_deleted += res.get("delete_count", 0)
        return {"delete_count": total_deleted}

    def delete_by_expr(
        self,
        filter_expr: str,
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if model is not None:
            return self._resolve_repo(model).delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return single_repo.delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                repo.delete_by_expr(filter_expr, user_id=user_id, partition_name=partition_name)
        return {"status": "SUCCESS"}

    def get_ghost_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch recorded audit events from global _forensic_audit_log ghost collection."""
        raw_items = self.client.query(
            collection_name=GHOST_AUDIT_LOG_COLLECTION,
            filter="",
            output_fields=["*"],
            limit=limit,
        )
        return raw_items

    def drop_collection(self, model_cls_or_name: Union[type[BaseModel], str]) -> None:
        """Drop a collection completely from Milvus by model class or collection name."""
        if isinstance(model_cls_or_name, str):
            col_name = model_cls_or_name
        else:
            col_name = get_model_tablename(model_cls_or_name)

        if self.client.has_collection(collection_name=col_name):
            self.client.drop_collection(collection_name=col_name)
            logger.info(f"Dropped collection '{col_name}' from Milvus")

        for key in list(self.repositories.keys()):
            if (
                key == model_cls_or_name
                or getattr(key, "__name__", "").lower() == str(col_name).lower()
            ):
                del self.repositories[key]

    def close(self) -> None:
        for repo in getattr(self, "repositories", {}).values():
            with contextlib.suppress(Exception):
                repo.close()
        shared_disp = getattr(self.client, "_wmilvus_audit_dispatcher", None)
        if shared_disp is not None:
            with contextlib.suppress(Exception):
                shared_disp.close()
            self.client._wmilvus_audit_dispatcher = None
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
        auto_flush: bool = False,
    ) -> BaseModel:
        return await asyncio.to_thread(
            self._repo.insert,
            record,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._repo.insert_batch,
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
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
        auto_flush: bool = False,
    ) -> BaseModel:
        return await asyncio.to_thread(
            self._repo.update,
            record_id,
            record,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def update_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._repo.update_batch,
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def flush(self) -> None:
        """Explicitly flush collection data to storage asynchronously."""
        await asyncio.to_thread(self._repo.flush)

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

    async def drop(self) -> None:
        """Drop this collection completely from the Milvus database asynchronously."""
        return await asyncio.to_thread(self._repo.drop)


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

    def _resolve_repo(self, model: Union[type[BaseModel], str]) -> AsyncCollectionRepository:
        """Resolve an async collection repository by model class or collection name."""
        if isinstance(model, type) and issubclass(model, BaseModel):
            if model in self.repositories:
                return self.repositories[model]
        elif isinstance(model, str):
            key = model.lower()
            if key in self._attr_repos:
                return self._attr_repos[key]
        raise CollectionError(f"No async repository registered for model '{model}'")

    async def insert(
        self,
        record: BaseModel,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.insert(
            record, user_id=user_id, partition_name=partition_name, auto_flush=auto_flush
        )

    async def insert_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = 1,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        if not records:
            return {"inserted_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return await repo.insert_batch(
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def get_all(
        self,
        model: Optional[Union[type[BaseModel], str, int]] = None,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if isinstance(model, int):
            limit = model
            model = None
        if model is not None:
            return await self._resolve_repo(model).get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
        combined: list[BaseModel] = []
        for repo in self.repositories.values():
            items = await repo.get_all(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
            combined.extend(items)
        return combined

    async def get_by_field(
        self,
        model: Optional[Union[type[BaseModel], str]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> Optional[BaseModel]:
        if model is not None and (
            (isinstance(model, type) and issubclass(model, BaseModel))
            or (isinstance(model, str) and model.lower() in self._attr_repos)
        ):
            return await self._resolve_repo(model).get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        if model is not None:
            kwargs["model"] = model
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        for repo in self.repositories.values():
            item = await repo.get_by_field(
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
            if item is not None:
                return item
        return None

    async def filter_by(
        self,
        model: Optional[Union[type[BaseModel], str]] = None,
        limit: int = 100,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = True,
        **kwargs: Any,
    ) -> list[BaseModel]:
        if model is not None and (
            (isinstance(model, type) and issubclass(model, BaseModel))
            or (isinstance(model, str) and model.lower() in self._attr_repos)
        ):
            return await self._resolve_repo(model).filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        if model is not None:
            kwargs["model"] = model
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.filter_by(
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
                **kwargs,
            )
        combined: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.filter_by(
                    limit=limit,
                    partition_names=partition_names,
                    output_fields=output_fields,
                    include_vector=include_vector,
                    **kwargs,
                )
                combined.extend(items)
        return combined

    async def search_similar(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if model is not None:
            return await self._resolve_repo(model).search_similar(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
                output_fields=output_fields,
                include_vector=include_vector,
            )
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.search_similar(
                    vector=vector,
                    top_k=top_k,
                    filter_expr=filter_expr,
                    search_params=search_params,
                    partition_names=partition_names,
                    output_fields=output_fields,
                    include_vector=include_vector,
                )
                results.extend(items)
        return results[:top_k]

    async def search_similar_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if model is not None:
            return await self._resolve_repo(model).search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.search_similar_with_scores(
                vector=vector,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
        matches: list[SearchMatch] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.search_similar_with_scores(
                    vector=vector,
                    top_k=top_k,
                    filter_expr=filter_expr,
                    search_params=search_params,
                    partition_names=partition_names,
                )
                matches.extend(items)
        matches.sort(key=lambda m: m.distance, reverse=True)
        return matches[:top_k]

    async def search_range(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[BaseModel]:
        if model is not None:
            return await self._resolve_repo(model).search_range(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.search_range(
                    vector=vector,
                    radius=radius,
                    top_k=top_k,
                    filter_expr=filter_expr,
                    search_params=search_params,
                    partition_names=partition_names,
                )
                results.extend(items)
        return results[:top_k]

    async def search_range_with_scores(
        self,
        vector: Union[list[float], np.ndarray],
        radius: float = 0.8,
        top_k: int = 10,
        model: Optional[Union[type[BaseModel], str]] = None,
        filter_expr: str = "",
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
    ) -> list[SearchMatch]:
        if model is not None:
            return await self._resolve_repo(model).search_range_with_scores(
                vector=vector,
                radius=radius,
                top_k=top_k,
                filter_expr=filter_expr,
                search_params=search_params,
                partition_names=partition_names,
            )
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
        matches: list[SearchMatch] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.search_range_with_scores(
                    vector=vector,
                    radius=radius,
                    top_k=top_k,
                    filter_expr=filter_expr,
                    search_params=search_params,
                    partition_names=partition_names,
                )
                matches.extend(items)
        matches.sort(key=lambda m: m.distance, reverse=True)
        return matches[:top_k]

    async def search_hybrid(
        self,
        vector: Union[list[float], np.ndarray],
        text_query: Optional[str] = None,
        text_field: Optional[str] = None,
        filter_expr: str = "",
        top_k: int = 5,
        model: Optional[Union[type[BaseModel], str]] = None,
        search_params: Optional[dict[str, Any]] = None,
        partition_names: Optional[list[str]] = None,
        ranker: Optional[Any] = None,
        output_fields: Optional[list[str]] = None,
        include_vector: bool = False,
    ) -> list[BaseModel]:
        if model is not None:
            return await self._resolve_repo(model).search_hybrid(
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
        results: list[BaseModel] = []
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                items = await repo.search_hybrid(
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
                results.extend(items)
        return results[:top_k]

    async def verify_schema(
        self, model: Optional[Union[type[BaseModel], str]] = None
    ) -> dict[str, Any]:
        if model is not None:
            return await self._resolve_repo(model).verify_schema()
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.verify_schema()
        results = {}
        for name_cls, repo in self.repositories.items():
            results[name_cls.__name__] = await repo.verify_schema()
        return {"status": "VALID", "collections": results}

    async def update(
        self,
        record_id: Union[str, int],
        record: BaseModel,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> BaseModel:
        model_cls = record.__class__
        repo = self[model_cls]
        return await repo.update(
            record_id,
            record,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def update_batch(
        self,
        records: list[BaseModel],
        batch_size: int = 1000,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
        auto_flush: bool = False,
    ) -> dict[str, Any]:
        if not records:
            return {"updated_count": 0, "batches_processed": 0}
        model_cls = records[0].__class__
        repo = self[model_cls]
        return await repo.update_batch(
            records,
            batch_size=batch_size,
            user_id=user_id,
            partition_name=partition_name,
            auto_flush=auto_flush,
        )

    async def flush(self, model: Optional[Union[type[BaseModel], str]] = None) -> None:
        """Explicitly flush collection data to storage asynchronously."""
        if model is not None:
            repo = self._resolve_repo(model)
            await repo.flush()
        elif len(self.repositories) == 1:
            repo = next(iter(self.repositories.values()))
            await repo.flush()
        else:
            for repo in self.repositories.values():
                await repo.flush()

    async def delete(
        self,
        record_id: Union[str, int],
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> None:
        if model is not None:
            await self._resolve_repo(model).delete(
                record_id, user_id=user_id, partition_name=partition_name
            )
            return
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            await single_repo.delete(record_id, user_id=user_id, partition_name=partition_name)
            return
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                await repo.delete(record_id, user_id=user_id, partition_name=partition_name)

    async def delete_batch(
        self,
        record_ids: list[Union[str, int]],
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if model is not None:
            return await self._resolve_repo(model).delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.delete_batch(
                record_ids, user_id=user_id, partition_name=partition_name
            )
        total_deleted = 0
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                res = await repo.delete_batch(
                    record_ids, user_id=user_id, partition_name=partition_name
                )
                total_deleted += res.get("delete_count", 0)
        return {"delete_count": total_deleted}

    async def delete_by_expr(
        self,
        filter_expr: str,
        model: Optional[Union[type[BaseModel], str]] = None,
        user_id: Optional[int] = None,
        partition_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if model is not None:
            return await self._resolve_repo(model).delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        if len(self.repositories) == 1:
            single_repo = next(iter(self.repositories.values()))
            return await single_repo.delete_by_expr(
                filter_expr, user_id=user_id, partition_name=partition_name
            )
        for repo in self.repositories.values():
            with contextlib.suppress(Exception):
                await repo.delete_by_expr(
                    filter_expr, user_id=user_id, partition_name=partition_name
                )
        return {"status": "SUCCESS"}

    async def drop_collection(self, model_cls_or_name: Union[type[BaseModel], str]) -> None:
        """Drop a collection completely from Milvus by model class or collection name asynchronously."""
        return await asyncio.to_thread(self._sync_client.drop_collection, model_cls_or_name)

    async def get_ghost_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync_client.get_ghost_audit_log, limit=limit)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_client.close)

    async def __aenter__(self) -> "AsyncWMilvus":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
