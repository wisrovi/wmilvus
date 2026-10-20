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

- **Pydantic Model Declarations** — Define collections as standard Pydantic models with vector annotations (`FieldVector`, `MetricType`, `IndexType`). `FieldVector` automatically defaults to `default_factory=list` to prevent missing field validation errors during deserialization.
- **Explicit Schema Generation & Auto-Migration** — Every attribute defined in your Pydantic model (`str`, `int`, `float`, `bool`, `list[str]`, `list[int]`, `dict`, `datetime`) is automatically mapped to dedicated Milvus columns (`VARCHAR`, `INT64`, `DOUBLE`, `BOOL`, `ARRAY`, `JSON`) rather than falling into generic dynamic fields (`$meta`). Existing collections automatically evolve by auto-adding missing columns (`add_collection_field`) upon client initialization.
- **Collection Lifecycle & Dropping (`drop`, `drop_collection`)** — Conveniently drop collections from Milvus via `db[Model].drop()` or `db.drop_collection(Model)` across synchronous and asynchronous clients.
- **Automatic Vector Retrieval & Preservation** — Query methods (`get_by_field`, `filter_by`) automatically include vector fields by default (`include_vector=True`). Calling `update()` with an empty vector automatically preserves the pre-existing vector from Milvus, eliminating dimension mismatch crashes on partial or metadata-only updates.
- **Dynamic Vector Fields & Auto-ID** — Native payload generation supporting custom vector attribute names (`embedding`, `face_vec`, etc.) and automated `INT64` primary key generation without requiring manual IDs.
- **Universal Primary Key Support** — Full compatibility with both auto-increment integer (`INT64`) and string (`VARCHAR`) primary keys across all CRUD, filtering, and search routines.
- **Collection Partitioning & Multi-Tenancy** — Route insertions, queries, and searches to specific Milvus partitions (`partition_name` / `partition_names`).
- **High-Scale Streaming Iterators (`iter_all`)** — Memory-efficient batch streaming across millions of vectors using native Milvus query iterators.
- **Typed Filter Query Builder (`filter_by`)** — Declarative keyword querying with comparison operators (`age__gte=18`, `status__in=[1, 2]`, `frames_ids__contains='f1'`, `dni='00000000P'`).
- **Vector Similarity & Range Search** — KNN similarity search (`search_similar`), similarity distance scores (`search_similar_with_scores`), and similarity range threshold search (`search_range`) with customizable search index parameters (`search_params`).
- **Batch & Expression Deletion** — Flexible record removal via single ID (`delete`), batch IDs (`delete_batch`), or scalar filter expressions (`delete_by_expr`).
- **Multi-Vector & Hybrid Search Rankers** — Dense + scalar hybrid search with direct support for Milvus rerankers (`RRFRanker`, `WeightedRanker`).
- **Bulk Vector Ingestion** — High-performance chunked bulk insertion (`insert_batch`) with automatic payload chunking and ID propagation.
- **Async Client Support (`AsyncWMilvus`)** — Full non-blocking async mirror with `async with` context manager for FastAPI and asyncio applications.
- **Schema Verification & Dimension Mismatch Detection** — Verification of Pydantic model vector dimensions against active Milvus collection schemas with explicit warning logging when existing server collections deviate from declared dimensions.
- **Single & Multi-Collection ORM Routing** — Automatic collection creation and full multi-table query support: call `db.get_all()` across all registered collections simultaneously, scope queries by model class (`db.get_all(Model)` or `db[Model].get_all()`), or access repositories directly by attribute (`db.modelname`).
- **Enterprise Ghost Audit Log (`_forensic_audit_log`)** — Global audit trail tracking `INSERT`, `UPDATE`, `SOFT_DELETE`, and `HARD_DELETE` operations for models inheriting from `ForensicModel` with safe JSON ISO-8601 datetime serialization.
- **Automated Collection Memory Loading** — Transparent auto-loading (`load_collection`) into memory for pre-existing or offline Milvus collections upon connection.
- **Flexible Client Instantiation** — Versatile initialization supporting `WMilvus(uri, models=[...])`, `WMilvus(models=[...], uri=...)`, and config dictionaries without rigid positional restrictions.
- **SQLite Vector Backup & Import (`wsqlite`)** — Native ORM-based backup export (`export_to_sqlite`) and restoration (`import_from_sqlite`) powered by `wsqlite`.
- **Dockerized Test Runner & Coverage** — Out-of-the-box support for isolated container testing (`run_tests_docker.sh`) and HTML coverage reports (`run_coverage.sh`).

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Vector Engine | Milvus 2.3+ |
| SDK Core | pymilvus 2.3+ |
| Backup Storage | wsqlite 1.5+ |
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
- [`examples/08_sqlite_backup/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/08_sqlite_backup/example.py) — SQLite vector backup export and import powered by `wsqlite`.
- [`examples/16_forensic_fields/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/16_forensic_fields/example.py) — Forensic field tracking (`create_by`, `create_in`, `update_by`).
- [`examples/17_multi_table/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/17_multi_table/example.py) — Multi-collection management & repository routing.
- [`examples/18_ghost_table_audit/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/18_ghost_table_audit/example.py) — Enterprise global audit trail (`_forensic_audit_log`).
- [`examples/19_auto_id_and_int64_pk/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/19_auto_id_and_int64_pk/example.py) — Auto-increment `INT64` primary key handling and model back-population.
- [`examples/20_query_builder_filter_by/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/20_query_builder_filter_by/example.py) — Fluent query builder with comparison operators (`__gte`, `__in`, `__lt`) and chained pagination.
- [`examples/21_streaming_iter_all/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/21_streaming_iter_all/example.py) — Memory-efficient streaming iterators (`iter_all`) across large collections.
- [`examples/22_partition_routing/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/22_partition_routing/example.py) — Creating collection partitions, targeted partition ingestion, and partition-scoped search.
- [`examples/23_batch_and_expr_deletion/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/23_batch_and_expr_deletion/example.py) — High-performance batch ID deletion and conditional deletion via Milvus boolean expressions.

## Running Tests

Execute unit tests inside an isolated Docker container:

```bash
./run_tests_docker.sh
```

Calculate code coverage locally:

```bash
./run_coverage.sh
```
