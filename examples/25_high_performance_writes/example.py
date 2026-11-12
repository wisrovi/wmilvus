"""High-performance writes example demonstrating WriteBuffer micro-batching and update_batch."""

from typing import List, Optional

from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus


class Face(BaseModel):
    id: Optional[int] = None
    user_name: str
    embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main():
    uri = "http://localhost:19530"

    # Option 1: WriteBuffer (Transparent micro-batching for 1-by-1 caller microservices)
    # Individual insert() and update() calls return immediately (<10ms) while
    # an internal daemon thread batches them into Milvus upserts every 25ms or 50 items.
    with WMilvus(
        Face,
        uri=uri,
        write_buffer=True,
        buffer_max_size=50,
        buffer_flush_interval=0.025,
    ) as db:
        for i in range(10):
            face = Face(user_name=f"User_{i}", embedding=[0.05 * i] * 128)
            # Returns quickly without waiting for full 600ms Milvus roundtrip
            db.insert(face)

        # Buffer is automatically flushed on context manager exit or explicit db.flush()
        db.flush()

    # Option 2: Native update_batch
    # Directly update multiple records in a single Milvus roundtrip
    with WMilvus(Face, uri=uri) as db:
        items_to_update = [
            Face(id=1, user_name="User_1_Updated", embedding=[0.1] * 128),
            Face(id=2, user_name="User_2_Updated", embedding=[0.2] * 128),
        ]
        db.update_batch(items_to_update)


if __name__ == "__main__":
    main()
