"""Example showing Single-Collection CRUD operations in WMilvus."""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class Person(BaseModel):
    """Pydantic model defining person schema and vector embedding."""

    id: str
    name: str
    age: int
    is_active: bool
    embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- Single-Collection CRUD Demonstration in WMilvus ---")

    # 1. Initialize WMilvus using context manager
    with WMilvus(Person, milvus_config) as db:
        # 2. Insert record
        person = Person(
            id="1",
            name="Juan Pérez",
            age=30,
            is_active=True,
            embedding=[0.1] * 128,
        )
        db.insert(person)
        print("Inserted Person!")

        # 3. Query all records
        all_people = db.get_all()
        print("All people in DB:", [p.name for p in all_people])

        # 4. Query by specific field
        user = db.get_by_field(name="Juan Pérez")
        if user:
            print(f"Found user by name: {user.name}, Age={user.age}")

        # 5. Update record
        updated_person = Person(
            id="1",
            name="Juan Pérez",
            age=31,
            is_active=False,
            embedding=[0.1] * 128,
        )
        db.update("1", updated_person)
        print("Updated record age to 31")

        # 6. Delete record
        db.delete("1")
        print("Deleted record ID=1")


if __name__ == "__main__":
    main()
