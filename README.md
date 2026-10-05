# wmilvus

<p align="center">
    <a href="https://pypi.org/project/wmilvus/">
        <img src="https://img.shields.io/pypi/v/wmilvus.svg" alt="PyPI version">
    </a>
    <a href="https://pypi.org/project/wmilvus/">
        <img src="https://img.shields.io/pypi/pyversions/wmilvus.svg" alt="Python versions">
    </a>
    <a href="https://github.com/wisrovi/wmilvus/blob/main/LICENSE">
        <img src="https://img.shields.io/pypi/l/wmilvus.svg" alt="License">
    </a>
</p>

**wmilvus** is an enterprise-grade, type-safe Milvus Vector Database client wrapper for Python designed with high-level Pydantic ORM model mapping, vector search, multi-collection routing, async support, and audit trail features.

## Key Features

- **Pydantic Model Declarations** — Define collections as standard Pydantic models with vector annotations (`FieldVector`, `MetricType`, `IndexType`).
- **Vector Similarity & Range Search** — KNN similarity search (`search_similar`), similarity distance scores (`search_similar_with_scores`), and similarity range threshold search (`search_range`).
- **Bulk Vector Ingestion** — High-performance chunked bulk insertion (`insert_batch`) with configurable payload size limits.
- **Async Client Support (`AsyncWMilvus`)** — Non-blocking async client with `async with` context manager for FastAPI and asyncio applications.
- **Schema Verification (`verify_schema`)** — Verification of Pydantic model vector dimensions against active Milvus collection schemas.
- **Hybrid Search (`search_hybrid`)** — Multi-modal search combining dense vector embeddings with text/scalar metadata filter expressions.
- **Single & Multi-Collection ORM Routing** — Automatic collection creation and routing by model class (`db[Model]`), attribute (`db.modelname`), or main client auto-routing (`db.insert(instance)`).
- **Enterprise Ghost Audit Log (`_forensic_audit_log`)** — Global audit trail tracking `INSERT`, `UPDATE`, `SOFT_DELETE`, and `HARD_DELETE` operations for models inheriting from `ForensicModel`.
- **Dockerized Test Runner & Coverage** — Out-of-the-box support for isolated container testing (`run_tests_docker.sh`) and HTML coverage reports (`run_coverage.sh`).

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Vector Engine | Milvus 2.3+ |
| SDK Core | pymilvus 2.3+ |
| Data Validation | Pydantic 2.x |
| Numerical Core | NumPy |
| Logging | Loguru |
| CLI Framework | Click |
| Testing Framework | pytest, pytest-cov |

## Quick Start

```python
from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus

class UserFace(BaseModel):
    id: str
    name: str
    face_embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)

# Context Manager Usage
with WMilvus(UserFace, uri="http://localhost:19530") as db:
    # 1. Insert Pydantic record
    user = UserFace(id="u_001", name="William Rodriguez", face_embedding=[0.1] * 128)
    db.insert(user)

    # 2. Vector Similarity Search returning typed Pydantic instances
    matches = db.search_similar(vector=[0.1] * 128, top_k=5)
    print(f"Top match: {matches[0].name}")

    # 3. Vector Search with Similarity Scores
    matches_with_scores = db.search_similar_with_scores(vector=[0.1] * 128, top_k=5)
    print(f"Score: {matches_with_scores[0].distance:.4f}")
```

## Async Quick Start

```python
import asyncio
from wmilvus import AsyncWMilvus, FieldVector, MetricType
from pydantic import BaseModel

class Telemetry(BaseModel):
    id: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)

async def main():
    async with AsyncWMilvus(Telemetry, uri="http://localhost:19530") as db:
        await db.insert(Telemetry(id="t1", vector=[0.5, 0.5, 0.0, 0.0]))
        results = await db.search_similar(vector=[0.5, 0.5, 0.1, 0.0], top_k=1)
        print(results[0].id)

asyncio.run(main())
```

## Organized Examples

The repository includes organized example scripts in `examples/`:

- [`examples/01_crud/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/01_crud/example.py) — Single-collection CRUD operations.
- [`examples/02_vector_search/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/02_vector_search/example.py) — Vector similarity KNN search, distance scores, and scalar metadata filtering.
- [`examples/03_range_search/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/03_range_search/example.py) — Similarity range threshold search (radius filtering).
- [`examples/04_batch_ingestion/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/04_batch_ingestion/example.py) — Bulk vector ingestion with automatic payload chunking (`insert_batch`).
- [`examples/05_async_client/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/05_async_client/example.py) — Async non-blocking client (`AsyncWMilvus`) for asyncio/FastAPI.
- [`examples/06_schema_verification/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/06_schema_verification/example.py) — Milvus collection schema verification (`verify_schema`).
- [`examples/07_hybrid_search/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/07_hybrid_search/example.py) — Hybrid similarity search (`search_hybrid`).
- [`examples/16_forensic_fields/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/16_forensic_fields/example.py) — Forensic field tracking (`create_by`, `create_in`, `update_by`).
- [`examples/17_multi_table/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/17_multi_table/example.py) — Multi-collection management & repository routing.
- [`examples/18_ghost_table_audit/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/18_ghost_table_audit/example.py) — Enterprise global audit trail (`_forensic_audit_log`).

## Running Tests

Execute unit tests inside an isolated Docker container:

```bash
./run_tests_docker.sh
```

Calculate code coverage locally:

```bash
./run_coverage.sh
```