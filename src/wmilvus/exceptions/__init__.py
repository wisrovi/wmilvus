"""WMilvus - Milvus Vector Database Client using Pydantic models."""

class WMilvusError(Exception):
    """Base exception for WMilvus errors."""
    pass


class ConnectionError(WMilvusError):
    """Exception raised for connection failures."""
    pass


class CollectionError(WMilvusError):
    """Exception raised for collection management errors."""
    pass


class ValidationError(WMilvusError):
    """Exception raised for schema or type validation failures."""
    pass


class VectorSearchError(WMilvusError):
    """Exception raised during vector search execution."""
    pass
