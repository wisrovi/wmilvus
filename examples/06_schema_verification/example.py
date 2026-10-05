"""Example showing Milvus Collection Schema Verification (verify_schema) in WMilvus.

This example demonstrates:
1. Verifying that an existing Milvus collection matches the bound Pydantic model's vector dimension.
2. Handling validation errors when schema dimensions mismatch.
"""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

milvus_uri = "http://localhost:19530"


class DocumentEmbedding(BaseModel):
    """Document vector embedding model."""

    id: str
    title: str
    vector: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main():
    print("--- Collection Schema Verification Demonstration in WMilvus ---")

    with WMilvus(DocumentEmbedding, uri=milvus_uri) as db:
        print("\n1. Verifying collection schema against Pydantic model...")
        schema_info = db.verify_schema()
        print(f"Schema Verification Result: {schema_info}")


if __name__ == "__main__":
    main()
