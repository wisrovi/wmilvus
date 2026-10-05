"""WMilvus - Simplified and resilient Milvus client wrapper for vector operations."""

from wmilvus.core.client import WMilvus
from wmilvus.exceptions import (
    CollectionError,
    ConnectionError,
    ValidationError,
    VectorSearchError,
    WMilvusError,
)
from wmilvus.types import SearchMatch, VectorRecord

__version__ = "0.1.0"

__all__ = [
    "WMilvus",
    "VectorRecord",
    "SearchMatch",
    "WMilvusError",
    "ConnectionError",
    "CollectionError",
    "ValidationError",
    "VectorSearchError",
]
