"""WMilvus - Milvus Vector Database ORM using Pydantic models."""

from wmilvus.core.client import (
    AsyncCollectionRepository,
    AsyncWMilvus,
    CollectionRepository,
    WMilvus,
)
from wmilvus.exceptions import (
    CollectionError,
    ConnectionError,
    ValidationError,
    VectorSearchError,
    WMilvusError,
)
from wmilvus.types import (
    FieldVector,
    ForensicModel,
    IndexType,
    MetricType,
    SearchMatch,
    VectorRecord,
)

__version__ = "0.1.0"

__all__ = [
    "WMilvus",
    "AsyncWMilvus",
    "CollectionRepository",
    "AsyncCollectionRepository",
    "FieldVector",
    "MetricType",
    "IndexType",
    "ForensicModel",
    "VectorRecord",
    "SearchMatch",
    "WMilvusError",
    "ConnectionError",
    "CollectionError",
    "ValidationError",
    "VectorSearchError",
]
