"""Core WMilvus ORM and Collection Client wrapper."""

from typing import Any, Dict, List, Optional, Type, Union
from loguru import logger
from pydantic import BaseModel
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from wmilvus.core.connection import ConnectionManager
from wmilvus.exceptions import CollectionError, VectorSearchError
from wmilvus.types import SearchResult, VectorFieldConfig


class WMilvus:
    """High-level, type-safe Milvus client with Pydantic model integration."""

    def __init__(
        self,
        model_class: Optional[Type[BaseModel]] = None,
        collection_name: Optional[str] = None,
        vector_config: Optional[VectorFieldConfig] = None,
        host: str = "localhost",
        port: int = 19530,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        alias: str = "default",
        **kwargs: Any,
    ) -> None:
        """Initialize WMilvus client instance."""
        self.model_class = model_class
        self.collection_name = collection_name or (model_class.__name__.lower() if model_class else "default_collection")
        self.vector_config = vector_config or VectorFieldConfig(dim=128)
        
        self.connection = ConnectionManager(
            alias=alias,
            host=host,
            port=port,
            uri=uri,
            token=token,
            **kwargs,
        )
        self._collection: Optional[Collection] = None

    def connect(self) -> "WMilvus":
        """Connect to Milvus instance and setup collection."""
        self.connection.connect()
        return self

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        self.connection.disconnect()

    def has_collection(self) -> bool:
        """Check if target collection exists."""
        return utility.has_collection(self.collection_name, using=self.connection.alias)

    def drop_collection(self) -> None:
        """Drop current collection if exists."""
        if self.has_collection():
            utility.drop_collection(self.collection_name, using=self.connection.alias)
            self._collection = None
            logger.info(f"Dropped collection '{self.collection_name}'")

    def create_collection(self, auto_id: bool = True) -> Collection:
        """Create Milvus collection based on vector_config and schema."""
        if self.has_collection():
            self._collection = Collection(self.collection_name, using=self.connection.alias)
            return self._collection

        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id)
        vector_field = FieldSchema(
            name=self.vector_config.name,
            dtype=DataType.FLOAT_VECTOR,
            dim=self.vector_config.dim,
        )
        schema = CollectionSchema(
            fields=[id_field, vector_field],
            description=f"Collection {self.collection_name} managed by WMilvus",
        )
        
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.connection.alias,
        )
        
        index_params = {
            "metric_type": self.vector_config.metric_type,
            "index_type": self.vector_config.index_type,
            "params": self.vector_config.params,
        }
        collection.create_index(
            field_name=self.vector_config.name,
            index_params=index_params,
        )
        
        self._collection = collection
        logger.info(f"Created collection '{self.collection_name}' with index {self.vector_config.index_type}")
        return self._collection

    def load(self) -> None:
        """Load collection into memory for searching."""
        if not self._collection:
            self.create_collection()
        self._collection.load()

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """Insert vectors and optional payloads into collection."""
        if not self._collection:
            self.create_collection()
        
        data = [vectors]
        mutation_res = self._collection.insert(data)
        self._collection.flush()
        return list(mutation_res.primary_keys)

    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """Search similar vectors in collection."""
        if not self._collection:
            self.create_collection()
            self.load()
            
        params = search_params or {"metric_type": self.vector_config.metric_type, "params": {"nprobe": 10}}
        
        try:
            results = self._collection.search(
                data=query_vectors,
                anns_field=self.vector_config.name,
                param=params,
                limit=top_k,
                output_fields=["id"],
            )
            
            formatted_results: List[List[SearchResult]] = []
            for raw_res in results:
                hit_list: List[SearchResult] = []
                for hit in raw_res:
                    hit_list.append(
                        SearchResult(
                            id=hit.id,
                            distance=hit.distance,
                            entity=hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else {},
                        )
                    )
                formatted_results.append(hit_list)
            return formatted_results
        except Exception as e:
            logger.error(f"Failed vector search on collection '{self.collection_name}': {e}")
            raise VectorSearchError(f"Search failed: {e}") from e
