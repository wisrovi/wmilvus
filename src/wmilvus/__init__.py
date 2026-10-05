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
from wmilvus.integrations.wpipe import WMilvusIngestStep
from wmilvus.integrations.wsqlite import export_to_sqlite, import_from_sqlite
from wmilvus.types import (
    FieldVector,
    ForensicModel,
    IndexType,
    MetricType,
    SearchMatch,
    VectorRecord,
    normalize_vector,
)

__version__ = "0.3.0"

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
    "normalize_vector",
    "WMilvusIngestStep",
    "export_to_sqlite",
    "import_from_sqlite",
    "WMilvusError",
    "ConnectionError",
    "CollectionError",
    "ValidationError",
    "VectorSearchError",
]
