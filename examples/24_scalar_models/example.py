"""Example showing scalar-only models without vector fields in WMilvus."""

from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class UserAccount(BaseModel):
    """Purely scalar/metadata model with NO vector fields."""

    id: str
    username: str
    email: str
    role: str = "member"
    is_active: bool = True


class UserFace(BaseModel):
    """Vector model with face biometric embedding."""

    id: str
    username: str
    embedding: list[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- Demonstrating Scalar-Only and Mixed Models in WMilvus ---")

    # WMilvus effortlessly supports both scalar-only and vector models together
    with WMilvus([UserAccount, UserFace], milvus_config) as db:
        # 1. Insert into scalar-only collection: no vector required!
        account = UserAccount(
            id="acc_001",
            username="william",
            email="william@example.com",
            role="admin",
            is_active=True,
        )
        db.insert(account)
        print("Inserted scalar UserAccount record (no vector field)!")

        # 2. Query scalar-only collection
        found = db[UserAccount].get_by_field(username="william")
        if found:
            print(f"Retrieved UserAccount: id={found.id}, username={found.username}, role={found.role}")
            print(f"Model attributes: {list(found.model_dump().keys())} (pure scalar, no dummy vector leakage)")

        # 3. Update scalar record
        found.role = "lead_architect"
        db[UserAccount].update("acc_001", found)
        updated = db[UserAccount].get_by_field(id="acc_001")
        if updated:
            print(f"Updated UserAccount role: {updated.role}")

        # 4. Insert into vector collection
        face = UserFace(id="face_001", username="william", embedding=[0.05] * 128)
        db.insert(face)
        print("Inserted UserFace vector record!")

        # 5. Search in vector collection
        matches = db[UserFace].search_similar(vector=[0.05] * 128, top_k=1)
        print(f"Top vector match in UserFace: {matches[0].username}")

        # 6. Combined get_all across all collections (scalar + vector)
        all_records = db.get_all()
        print(f"Total records retrieved across all collections: {len(all_records)}")


if __name__ == "__main__":
    main()
