"""Basic usage example for WMilvus."""

from pydantic import BaseModel
from wmilvus import WMilvus, VectorFieldConfig


class ImageVector(BaseModel):
    image_id: str
    description: str


def main() -> None:
    # 1. Define vector configuration (e.g., 128 dimension vectors)
    config = VectorFieldConfig(dim=128, metric_type="COSINE", index_type="HNSW")

    # 2. Instantiate client
    client = WMilvus(
        model_class=ImageVector,
        vector_config=config,
        host="localhost",
        port=19530,
    )

    print(f"Initialized WMilvus for collection: {client.collection_name}")


if __name__ == "__main__":
    main()
