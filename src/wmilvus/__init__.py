"""WMilvus - Milvus Vector Database ORM using Pydantic models."""

from pymilvus import RRFRanker, WeightedRanker

from wmilvus.core.client import (
    AsyncCollectionRepository,
    AsyncWMilvus,
    CollectionRepository,
    WMilvus,
    build_filter_expr,
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

__version__ = "0.4.4"

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
    "build_filter_expr",
    "RRFRanker",
    "WeightedRanker",
    "WMilvusError",
    "ConnectionError",
    "CollectionError",
    "ValidationError",
    "VectorSearchError",
]
