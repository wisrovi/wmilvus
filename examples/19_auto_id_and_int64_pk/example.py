"""Example demonstrating Auto-ID and INT64 Primary Keys in WMilvus."""

from typing import List, Optional
from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class FaceEmbedding(BaseModel):
    """Pydantic model with auto-increment INT64 primary key."""

    __tablename__ = "FaceEmbeddingsAutoId"

    id: Optional[int] = Field(None, description="Primary key AUTOINCREMENT INT64")
    name: str = Field(..., description="Subject name")
    embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- Auto-ID and INT64 Primary Keys in WMilvus ---")

    with WMilvus(FaceEmbedding, milvus_config) as db:
        # 1. Insert record without specifying an ID (Milvus auto-assigns INT64)
        new_face = FaceEmbedding(
            name="Alice Smith",
            embedding=[0.05] * 128,
        )
        saved_face = db.insert(new_face)
        print(f"Inserted record with auto-assigned INT64 ID: {saved_face.id}")

        # 2. Query record by the assigned integer primary key
        if saved_face.id is not None:
            fetched_face = db.get_by_id(saved_face.id)
            if fetched_face:
                print(f"Fetched by INT64 ID ({saved_face.id}): {fetched_face.name}")

        # 3. Vector similarity search returns models with correct INT64 primary keys
        matches = db.search_similar(vector=[0.05] * 128, top_k=1)
        if matches:
            print(f"Top vector match: ID={matches[0].id}, Name={matches[0].name}")


if __name__ == "__main__":
    main()
