"""Example showing AsyncWMilvus Client Usage in Asyncio / FastAPI applications.

This example demonstrates:
1. Using async with AsyncWMilvus(...) as db for non-blocking vector ORM operations.
2. Async insert, async similarity search, and async batch operations.
"""

import asyncio
from typing import List
from pydantic import BaseModel
from wmilvus import AsyncWMilvus, FieldVector, MetricType

milvus_uri = "http://localhost:19530"


class AsyncSensorVector(BaseModel):
    """IoT Sensor telemetry vector embedding model."""

    id: str
    location: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)


async def main():
    print("--- AsyncWMilvus Non-Blocking Client Demonstration ---")

    async with AsyncWMilvus(AsyncSensorVector, uri=milvus_uri) as db:
        print("\n1. Async inserting telemetry vectors...")
        records = [
            AsyncSensorVector(id="s_101", location="Building_A", vector=[0.5, 0.5, 0.0, 0.0]),
            AsyncSensorVector(id="s_102", location="Building_B", vector=[0.1, 0.9, 0.0, 0.0]),
        ]
        for rec in records:
            await db.insert(rec)
        print("Async insertion completed!")

        # Short pause to ensure segment visibility in rapid test suites
        await asyncio.sleep(0.3)

        print("\n2. Async similarity search...")
        matches = await db.search_similar(vector=[0.5, 0.5, 0.1, 0.0], top_k=1)
        if matches:
            print(f"Async top match: ID={matches[0].id}, Location={matches[0].location}")
        else:
            print("No matches returned in search.")


if __name__ == "__main__":
    asyncio.run(main())
