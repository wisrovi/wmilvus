"""Example usage of WMilvus wrapper."""

import numpy as np
from wmilvus import VectorRecord, WMilvus


def main() -> None:
    # 1. Initialize client
    milvus = WMilvus(uri="http://localhost:19530")

    # 2. Ensure collection exists (e.g. 512-dim visual embeddings)
    collection = "visual_embeddings_v1"
    milvus.ensure_collection(collection_name=collection, dimension=512, metric_type="COSINE")

    # 3. Upsert a batch of vectors with metadata
    sample_vector = np.random.rand(512).astype(np.float32).tolist()
    records = [
        VectorRecord(
            id="frame_00104",
            vector=sample_vector,
            metadata={"camera_id": "cam_entry_north", "confidence": 0.94, "label": "forklift"},
        )
    ]
    upsert_res = milvus.upsert_batch(collection_name=collection, records=records)
    print(f"Upsert result: {upsert_res}")

    # 4. Perform vector similarity search
    query = np.random.rand(512).astype(np.float32)
    results = milvus.search_similar(
        collection_name=collection,
        query_vector=query,
        top_k=3,
        filter_expr='label == "forklift"',
    )

    for match in results:
        print(f"ID: {match.id} | Score: {match.distance:.4f} | Meta: {match.metadata}")


if __name__ == "__main__":
    main()
