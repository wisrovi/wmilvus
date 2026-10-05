"""WMilvus - Milvus Vector Database Client using Pydantic models."""

from wmilvus.core.client import WMilvus
from wmilvus.core.connection import ConnectionManager
from wmilvus.exceptions import (
    CollectionError,
    ConnectionError,
    ValidationError,
    VectorSearchError,
    WMilvusError,
)
from wmilvus.types import SearchResult, VectorFieldConfig

__version__ = "0.1.0"

__all__ = [
    "WMilvus",
    "ConnectionManager",
    "VectorFieldConfig",
    "SearchResult",
    "WMilvusError",
    "ConnectionError",
    "CollectionError",
    "ValidationError",
    "VectorSearchError",
]
