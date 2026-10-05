"""Example showing Hybrid Similarity Search (Dense Vector + Text/Scalar Filter) in WMilvus.

This example demonstrates:
1. Indexing multi-modal entities with vector embeddings and text attributes.
2. Performing hybrid search combining dense similarity with text matching expressions (search_hybrid).
"""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

milvus_uri = "http://localhost:19530"


class ArticleVector(BaseModel):
    """Knowledge base article model."""

    id: str
    title: str
    author: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)


def main():
    print("--- Hybrid Search (Dense Vector + Text Filtering) Demonstration in WMilvus ---")

    with WMilvus(ArticleVector, uri=milvus_uri) as db:
        print("\n1. Indexing articles...")
        articles = [
            ArticleVector(id="art_01", title="Deep Learning with Milvus", author="William Rodriguez", vector=[0.9, 0.1, 0.0, 0.1]),
            ArticleVector(id="art_02", title="Vector Databases in Production", author="Alice Smith", vector=[0.85, 0.15, 0.05, 0.0]),
            ArticleVector(id="art_03", title="Introduction to Python Asyncio", author="William Rodriguez", vector=[0.1, 0.8, 0.5, 0.2]),
        ]
        for a in articles:
            db.insert(a)

        query_vector = [0.89, 0.11, 0.01, 0.08]
        print(f"\n2. Performing hybrid search for author='William Rodriguez' and dense similarity...")

        matches = db.search_hybrid(
            vector=query_vector,
            text_field="author",
            text_query="William",
            top_k=5,
        )

        print(f"Hybrid search returned {len(matches)} match(es):")
        for m in matches:
            print(f"  Article ID={m.id}, Title='{m.title}', Author='{m.author}'")


if __name__ == "__main__":
    main()
