"""Milvus entity models, field vector annotations, and result types."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def FieldVector(
    dim: int = 128,
    metric_type: str = "COSINE",
    index_type: str = "HNSW",
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Helper field metadata annotation for vector embedding attributes in Pydantic models.

    Args:
        dim: Vector dimension length.
        metric_type: Similarity metric type ('COSINE', 'L2', 'IP').
        index_type: Index algorithm ('HNSW', 'IVF_FLAT', 'FLAT').
        params: Additional index building parameters.

    Returns:
        Pydantic Field instance with vector metadata attached.
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {})
    if not isinstance(json_schema_extra, dict):
        json_schema_extra = {}
    
    json_schema_extra.update({
        "is_vector": True,
        "dim": dim,
        "metric_type": metric_type,
        "index_type": index_type,
        "params": params or {"M": 16, "efConstruction": 200},
    })
    
    return Field(json_schema_extra=json_schema_extra, **kwargs)


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


class ForensicModel(BaseModel):
    """Base Pydantic model with forensic audit fields for WMilvus.

    Inheriting from this model automatically enables forensic auditing for Milvus collections.
    """

    create_by: Optional[int] = 1
    create_in: Optional[datetime] = None
    update_by: Optional[int] = None
    update_in: Optional[datetime] = None
    delete_by: Optional[int] = None
    delete_in: Optional[datetime] = None
    status: int = 1
