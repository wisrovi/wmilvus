"""Milvus entity models, field vector annotations, enums, and result types."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Vector similarity metric types for Milvus indexing and search."""

    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"


class IndexType(str, Enum):
    """Vector index algorithm types for Milvus."""

    HNSW = "HNSW"
    IVF_FLAT = "IVF_FLAT"
    FLAT = "FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"


def FieldVector(
    dim: int = 128,
    metric_type: Union[str, MetricType] = MetricType.COSINE,
    index_type: Union[str, IndexType] = IndexType.HNSW,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Helper field metadata annotation for vector embedding attributes in Pydantic models.

    Args:
        dim: Vector dimension length.
        metric_type: Similarity metric type (MetricType.COSINE, MetricType.L2, MetricType.IP, etc.).
        index_type: Index algorithm (IndexType.HNSW, IndexType.IVF_FLAT, IndexType.FLAT, etc.).
        params: Additional index building parameters.

    Returns:
        Pydantic Field instance with vector metadata attached.
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {})
    if not isinstance(json_schema_extra, dict):
        json_schema_extra = {}

    m_type = metric_type.value if isinstance(metric_type, MetricType) else str(metric_type)
    idx_type = index_type.value if isinstance(index_type, IndexType) else str(index_type)

    json_schema_extra.update({
        "is_vector": True,
        "dim": dim,
        "metric_type": m_type,
        "index_type": idx_type,
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
