"""Example showing Multi-Collection management using WMilvus.

This example demonstrates:
1. Multi-collection initialization passing a list of Pydantic models.
2. Dictionary indexing (db[UserFace]) and direct attribute access (db.productimage).
3. Automatic routing of insert() operations.
"""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, ForensicModel, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class UserFace(ForensicModel):
    """User face embedding model."""

    __tablename__ = "users_face"
    id: str
    name: str
    email: str
    face_vec: List[float] = FieldVector(dim=128)


class ProductImage(BaseModel):
    """Catalog product image embedding model."""

    id: str
    title: str
    price: float
    image_vec: List[float] = FieldVector(dim=64)


def main() -> None:
    print("--- Initializing WMilvus in Multi-Collection Mode ---")

    with WMilvus([UserFace, ProductImage], milvus_config) as db:
        user = UserFace(id="u_001", name="William Rodriguez", email="william@example.com", face_vec=[0.1] * 128)
        product = ProductImage(id="p_101", title="Antigravity Vision AI", price=199.99, image_vec=[0.5] * 64)

        # Method A: Indexing by class (Type-safe)
        db[UserFace].insert(user)

        # Method B: Direct attribute access in lowercase
        db.productimage.insert(product)

        # Method C: Auto-routing insert via main db instance
        db.insert(user)

        print("Insertion complete across all Milvus collections!")


if __name__ == "__main__":
    main()
