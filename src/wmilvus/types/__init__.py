"""Milvus collection schema definitions and vector metadata models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VectorFieldConfig(BaseModel):
    """Configuration for vector fields in Milvus collections."""

    name: str = Field(default="vector", description="Name of the vector field")
    dim: int = Field(..., description="Vector dimension")
    metric_type: str = Field(default="COSINE", description="Metric type (COSINE, L2, IP)")
    index_type: str = Field(default="HNSW", description="Index type (HNSW, IVF_FLAT, FLAT)")
    params: Dict[str, Any] = Field(default_factory=lambda: {"M": 16, "efConstruction": 200})


class SearchResult(BaseModel):
    """Result item from a vector similarity search."""

    id: Any = Field(..., description="Entity ID")
    distance: float = Field(..., description="Distance / similarity score")
    entity: Dict[str, Any] = Field(default_factory=dict, description="Retrieved payload fields")
