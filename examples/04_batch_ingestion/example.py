"""Example showing Bulk Vector Ingestion with Automatic Chunking (insert_batch) in WMilvus.

This example demonstrates:
1. Generating a batch of 2,500 vector records.
2. Bulk inserting records into Milvus using insert_batch with automatic chunking (batch_size=1000).
3. Verifying ingestion counts across all chunked payloads.
"""

from typing import List
import numpy as np
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

milvus_uri = "http://localhost:19530"


class FrameEmbedding(BaseModel):
    """Video frame embedding model for large scale ingestion."""

    id: str
    camera_id: str
    vector: List[float] = FieldVector(dim=8, metric_type=MetricType.COSINE)


def main():
    print("--- Bulk Vector Ingestion (insert_batch) Demonstration in WMilvus ---")

    with WMilvus(FrameEmbedding, uri=milvus_uri) as db:
        print("\n1. Generating 2,500 synthetic vector records...")
        records = [
            FrameEmbedding(
                id=f"frame_{idx:05d}",
                camera_id=f"cam_{idx % 5}",
                vector=np.random.rand(8).tolist(),
            )
            for idx in range(2500)
        ]

        print("\n2. Executing bulk insertion with batch_size=1000...")
        result = db.insert_batch(records, batch_size=1000)
        print(f"Batch Ingestion Result: {result}")

        # Query all count
        stored_items = db.get_all(limit=3000)
        print(f"Successfully retrieved {len(stored_items)} records from Milvus!")


if __name__ == "__main__":
    main()
