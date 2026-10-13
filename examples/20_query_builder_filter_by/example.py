"""Example demonstrating the Declarative Query Builder (filter_by) in WMilvus."""

from typing import List
from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class EmployeeProfile(BaseModel):
    """Pydantic model for employee vector embeddings with structured metadata."""

    __tablename__ = "EmployeeProfiles"

    id: str = Field(..., description="Unique employee badge ID")
    name: str = Field(..., description="Full employee name")
    department: str = Field(..., description="Department name")
    age: int = Field(..., description="Employee age")
    years_experience: int = Field(..., description="Years of professional experience")
    embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- Declarative Query Builder (filter_by) in WMilvus ---")

    with WMilvus(EmployeeProfile, milvus_config) as db:
        # Ingest sample profiles
        samples = [
            EmployeeProfile(
                id="emp_01",
                name="Alice Johnson",
                department="AI Research",
                age=29,
                years_experience=5,
                embedding=[0.1] * 128,
            ),
            EmployeeProfile(
                id="emp_02",
                name="Bob Miller",
                department="AI Research",
                age=35,
                years_experience=10,
                embedding=[0.2] * 128,
            ),
            EmployeeProfile(
                id="emp_03",
                name="Charlie Davis",
                department="Platform",
                age=24,
                years_experience=2,
                embedding=[0.3] * 128,
            ),
        ]
        for s in samples:
            db.insert(s)

        # 1. Filter with exact match and comparison operators (__gte, __in)
        ai_veterans = (
            db.filter_by(
                department="AI Research",
                years_experience__gte=5,
            )
            .limit(10)
            .all()
        )
        print(f"Found {len(ai_veterans)} AI veterans (experience >= 5):")
        for emp in ai_veterans:
            print(f" - {emp.name} ({emp.department}), Age={emp.age}, Exp={emp.years_experience}")

        # 2. Retrieve first matching record
        first_young = db.filter_by(age__lt=30).first()
        if first_young:
            print(f"First match under 30: {first_young.name}, Age={first_young.age}")


if __name__ == "__main__":
    main()
