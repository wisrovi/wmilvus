"""Multi-table / Multi-collection Pydantic ORM demonstration for WMilvus."""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, ForensicModel, WMilvus


class UserFace(ForensicModel):
    """User account face embedding model with forensic auditing enabled."""

    __tablename__ = "users_face"
    id: str
    name: str
    email: str
    face_vec: List[float] = FieldVector(dim=128, metric_type="COSINE")


class ProductImage(BaseModel):
    """Catalog product image embedding model."""

    id: str
    title: str
    price: float
    image_vec: List[float] = FieldVector(dim=64, metric_type="COSINE")


def main() -> None:
    """Execute multi-collection Pydantic ORM demonstration."""
    print("--- Initializing WMilvus in Multi-Collection Mode ---")

    milvus_config = {
        "uri": "http://localhost:19530",
        "token": "",
        "db_name": "default",
    }

    # 1. Initialize WMilvus with a list of Pydantic models
    db = WMilvus([UserFace, ProductImage], db_config=milvus_config)

    user = UserFace(
        id="u_001",
        name="William Rodriguez",
        email="william@example.com",
        face_vec=[0.1] * 128,
    )
    product = ProductImage(
        id="p_101",
        title="Antigravity Vision AI",
        price=199.99,
        image_vec=[0.5] * 64,
    )

    # Method A: Indexing by class (Type-safe)
    db[UserFace].insert(user)

    # Method B: Direct attribute access in lowercase
    db.productimage.insert(product)

    # Method C: Auto-routing insert via main db instance
    db.insert(user)

    print("Insertion complete across all Milvus collections!")


if __name__ == "__main__":
    main()
