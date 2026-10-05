"""Example showing Vector Similarity Search (KNN/ANN) in WMilvus.

This example demonstrates:
1. Defining a Pydantic model with vector embeddings using FieldVector and MetricType.COSINE.
2. Context manager usage with 'with WMilvus(...) as db:' for automatic resource cleanup.
3. Inserting multiple embedding vectors into Milvus.
4. Performing top-k similarity search returning typed Pydantic models.
5. Performing vector search with similarity distance scores.
6. Combining vector search with scalar Boolean metadata filtering.
"""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

# Milvus connection URI
milvus_uri = "http://localhost:19530"


class FaceEmbedding(BaseModel):
    """Face biometric embedding representation model using MetricType enum."""

    id: str
    person_name: str
    category: str
    embedding: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)


def main():
    """Execute vector similarity search demonstration."""
    print("--- Vector Similarity Search Demonstration in WMilvus ---")

    # Clean context manager usage - no manual db.close() required!
    with WMilvus(FaceEmbedding, uri=milvus_uri) as db:
        # 1. Insert sample vector embeddings into Milvus
        print("\n1. Indexing face embedding vectors...")
        records = [
            FaceEmbedding(
                id="f_001",
                person_name="William Rodriguez",
                category="VIP",
                embedding=[0.9, 0.1, 0.0, 0.1],
            ),
            FaceEmbedding(
                id="f_002",
                person_name="Alice Smith",
                category="VIP",
                embedding=[0.85, 0.15, 0.05, 0.0],
            ),
            FaceEmbedding(
                id="f_003",
                person_name="Bob Jones",
                category="Standard",
                embedding=[0.1, 0.8, 0.5, 0.2],
            ),
            FaceEmbedding(
                id="f_004",
                person_name="Charlie Brown",
                category="Standard",
                embedding=[0.05, 0.75, 0.6, 0.1],
            ),
        ]

        for rec in records:
            db.insert(rec)
        print("Successfully indexed 4 face vectors!")

        # 2. Query vector search (KNN search)
        query_vector = [0.92, 0.08, 0.01, 0.05]
        print(f"\n2. Searching top-2 nearest neighbors for query vector: {query_vector}...")

        # Method A: Direct search returning typed Pydantic instances
        nearest_models = db.search_similar(vector=query_vector, top_k=2)
        print("\n[Method A] Top Matches (Typed Pydantic Models):")
        for idx, model in enumerate(nearest_models, 1):
            print(f"  Match #{idx}: {model.person_name} (Category: {model.category}, ID: {model.id})")

        # Method B: Search returning similarity scores
        matches_with_scores = db.search_similar_with_scores(vector=query_vector, top_k=2)
        print("\n[Method B] Top Matches with Similarity Distance Scores:")
        for idx, match in enumerate(matches_with_scores, 1):
            print(f"  Match #{idx}: ID={match.id}, Score={match.distance:.4f}, Name={match.metadata.get('person_name')}")

        # 3. Vector search combined with scalar metadata filtering
        print("\n3. Vector search filtered by scalar metadata (category == 'VIP')...")
        filtered_results = db.search_similar(
            vector=query_vector,
            top_k=2,
            filter_expr='category == "VIP"',
        )
        print(f"Found {len(filtered_results)} VIP match(es):")
        for model in filtered_results:
            print(f"  VIP Match: {model.person_name} (ID: {model.id})")


if __name__ == "__main__":
    main()
