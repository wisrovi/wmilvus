"""Milvus entity models and similarity search types."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Data transfer object representing a vector entity for indexing."""

    id: str = Field(description="Unique identifier for the vector record")
    vector: List[float] = Field(description="Dense vector embedding representation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata associated with the record",
    )


class SearchMatch(BaseModel):
    """Result item returned from a vector similarity search."""

    id: str = Field(description="Identifier of the matched entity")
    distance: float = Field(description="Similarity distance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieved entity metadata and attributes",
    )
