"""Example showing Similarity Range Search (Radius Search) in WMilvus.

This example demonstrates:
1. Defining a Pydantic model with vector embeddings.
2. Performing vector range search filtering by minimum similarity score (radius threshold >= 0.90).
3. Obtaining SearchMatch objects with similarity scores within the range.
"""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

milvus_uri = "http://localhost:19530"


class ProductEmbedding(BaseModel):
    """Product vector embedding model."""

    id: str
    name: str
    category: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)


def main():
    print("--- Similarity Range Search (Radius Threshold) Demonstration in WMilvus ---")

    with WMilvus(ProductEmbedding, uri=milvus_uri) as db:
        print("\n1. Indexing product vectors...")
        products = [
            ProductEmbedding(id="p_001", name="Smart Watch Alpha", category="Electronics", vector=[0.9, 0.1, 0.0, 0.1]),
            ProductEmbedding(id="p_002", name="Smart Watch Beta", category="Electronics", vector=[0.88, 0.12, 0.02, 0.08]),
            ProductEmbedding(id="p_003", name="Running Shoes", category="Footwear", vector=[0.1, 0.8, 0.5, 0.2]),
        ]
        for p in products:
            db.insert(p)

        query_vec = [0.91, 0.09, 0.01, 0.09]
        print(f"\n2. Searching for products with COSINE similarity >= 0.95 to query vector {query_vec}...")

        matches_with_scores = db.search_range_with_scores(vector=query_vec, radius=0.95, top_k=5)
        print(f"Found {len(matches_with_scores)} match(es) above 0.95 similarity score:")
        for m in matches_with_scores:
            print(f"  Match ID={m.id}, Score={m.distance:.4f}, Name={m.metadata.get('name')}")


if __name__ == "__main__":
    main()
