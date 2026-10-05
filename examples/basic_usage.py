"""Basic usage example for WMilvus."""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, WMilvus


class ImageVector(BaseModel):
    id: str
    image_id: str
    description: str
    vector: List[float] = FieldVector(dim=128, metric_type="COSINE")


def main() -> None:
    milvus_config = {
        "uri": "http://localhost:19530",
        "token": "",
        "db_name": "default",
    }

    db = WMilvus(ImageVector, db_config=milvus_config)
    print(f"Initialized WMilvus for collection: {ImageVector.__name__.lower()}")


if __name__ == "__main__":
    main()
