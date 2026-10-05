# Getting Started with `wmilvus`

This guide gets you up and running with `wmilvus` in minutes.

---

## 1. Installation

Install `wmilvus` from PyPI:

```bash
pip install wmilvus
```

To include development and backup dependencies:

```bash
pip install "wmilvus[dev]"
```

---

## 2. Defining Pydantic Vector Models

Collections in `wmilvus` are defined as standard Pydantic models with vector annotations provided by `FieldVector`:

```python
from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, IndexType

class FaceProfile(BaseModel):
    id: str
    name: str
    department: str
    embedding: List[float] = FieldVector(
        dim=128,
        metric_type=MetricType.COSINE,
        index_type=IndexType.HNSW
    )
```

---

## 3. Synchronous Client Usage (`WMilvus`)

Use `WMilvus` with a `with` context manager for automatic connection and cleanup:

```python
from wmilvus import WMilvus

# Connect to local or remote Milvus server
with WMilvus(FaceProfile, uri="http://localhost:19530") as db:
    # Insert record
    profile = FaceProfile(
        id="usr_101",
        name="William Rodriguez",
        department="AI Architecture",
        embedding=[0.05] * 128
    )
    db.insert(profile)

    # Search similar vectors
    query_vector = [0.05] * 128
    results = db.search_similar(vector=query_vector, top_k=5)

    print(f"Matched Profile: {results[0].name}")
```

---

## 4. Asynchronous Client Usage (`AsyncWMilvus`)

For high-concurrency FastAPI or asyncio applications, use `AsyncWMilvus`:

```python
import asyncio
from wmilvus import AsyncWMilvus, FieldVector, MetricType
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    id: str
    text: str
    vector: List[float] = FieldVector(dim=384, metric_type=MetricType.COSINE)

async def run_async():
    async with AsyncWMilvus(DocumentChunk, uri="http://localhost:19530") as db:
        chunk = DocumentChunk(id="c1", text="Milvus ORM", vector=[0.1] * 384)
        await db.insert(chunk)

        matches = await db.search_similar(vector=[0.1] * 384, top_k=3)
        print(f"Top async match: {matches[0].text}")

asyncio.run(run_async())
```
