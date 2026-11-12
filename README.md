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
- **High-Performance Non-Blocking Writes & Explicit Flushing** — In adherence to official Milvus architecture best practices, automatic forced `flush()` calls after every `insert`, `insert_batch`, and `update` have been removed (`auto_flush=False` by default). This eliminates severe I/O latency bottlenecks, query node memory saturation, and tiny segment fragmentation caused by sealing growing segments prematurely. If immediate read-after-write consistency is needed, Milvus's native `consistency_level="Strong"` or `"Session"` is recommended. Manual flushing is available via `db.flush()` or `db[Model].flush()` when required administratively.
- **Asynchronous Forensic Audit Dispatcher (`ForensicAuditDispatcher`)** — Decouples audit trail writes (`_forensic_audit_log`) into a background daemon queue with micro-batching. This reduces single-item write latency on `ForensicModel` by 100-220ms per operation without sacrificing audit guarantees (synchronized on process termination, client close, or manual flush).
- **Transparent Write-Behind Buffer (`write_buffer=True`)** — High-performance write buffer for microservices that issue single-record inserts or updates. Operations return immediately (<10ms) while a background daemon coalesces writes into sub-second Milvus upsert batches (e.g. 25ms window or up to 50 items), cutting round-trip latency by over 80%.
- **High-Speed Batch Updates (`update_batch`)** — Update multiple records in a single round-trip via `db.update_batch([records])` across synchronous and asynchronous clients, with vector preservation and batched audit trails.
- **In-Memory Ghost Collection Caching** — Verifications of collection existence for `_forensic_audit_log` are memoized in memory, eliminating redundant gRPC `has_collection` calls on every write.
- **Selective Scalar Querying for Updates** — Pre-update queries for `ForensicModel` audit snapshots automatically exclude 512-dimension vectors when the incoming record already supplies the vector, saving bandwidth and network overhead.
- **Explicit `data_before` Injection & In-Memory LRU Entity Cache** — Update operations (`update()`, `update_batch()`) allow direct injection of previous state (`data_before=...`) and leverage an internal LRU cache populated by `insert()` and `get_by_field()`. This avoids synchronous Milvus read queries entirely, plummeting single-record update latency down to `< 5ms`.
- **Bulk Vector Ingestion** — High-performance chunked bulk insertion (`insert_batch`) with automatic payload chunking and ID propagation.
- **Async Client Support (`AsyncWMilvus`)** — Full non-blocking async mirror with `async with` context manager for FastAPI and asyncio applications.
- **Schema Verification & Dimension Mismatch Detection** — Verification of Pydantic model vector dimensions against active Milvus collection schemas with explicit warning logging when existing server collections deviate from declared dimensions.
- **Optional Vector Fields & Scalar-Only Collections** — Collections do not strictly require vector fields. Pure scalar or metadata models (e.g., users, categories, configurations, relational entities) can be defined without any vector fields. WMilvus transparently satisfies Milvus's internal vector requirement under the hood with lightweight synthetic dummy vectors, enabling full type-safe CRUD, filtering, streaming, and multi-table querying without imposing vectors on your models.
- **Single & Multi-Collection ORM Routing** — Automatic collection creation and full multi-table query support: call `db.get_all()` across all registered collections simultaneously, scope queries by model class (`db.get_all(Model)` or `db[Model].get_all()`), or access repositories directly by attribute (`db.modelname`).
- **Enterprise Ghost Audit Log (`_forensic_audit_log`)** — Global audit trail tracking `INSERT`, `UPDATE`, `SOFT_DELETE`, and `HARD_DELETE` operations for models inheriting from `ForensicModel` with safe JSON ISO-8601 datetime serialization.
- **Automated Collection Memory Loading** — Transparent auto-loading (`load_collection`) into memory for pre-existing or offline Milvus collections upon connection.
- **Flexible Client Instantiation** — Versatile initialization supporting `WMilvus(uri, models=[...])`, `WMilvus(models=[...], uri=...)`, and config dictionaries without rigid positional restrictions.
- **SQLite Vector Backup & Import (`wsqlite`)** — Native ORM-based backup export (`export_to_sqlite`) and restoration (`import_from_sqlite`) powered by `wsqlite`.
- **Dockerized Test Runner & Coverage** — Out-of-the-box support for isolated container testing (`run_tests_docker.sh`) and HTML coverage reports (`run_coverage.sh`).

## Key Technologies and Libraries

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
| Packaging & Build | setuptools>=61.0, wheel |
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

## Write Performance & Consistency

By default in `wmilvus>=0.4.6`, automatic `flush()` calls during `insert()`, `insert_batch()`, and `update()` are disabled (`auto_flush=False`). Calling `flush()` after every write is strongly discouraged by official Milvus documentation because it forcibly seals growing segments in memory, writes tiny files to storage, and triggers expensive background compactions.

### High-Throughput Write-Behind Buffer (`write_buffer=True`)

When microservices are architecturally constrained to call `insert()` or `update()` one record at a time, each standalone network call against Milvus can incur significant network and proxy overhead (typically ~600ms per update due to Milvus's tombstone deletion and indexing mechanism).

To solve this without modifying calling microservices:
```python
# Enable transparent write-behind micro-batching
with WMilvus(
    UserFace,
    uri="http://localhost:19530",
    write_buffer=True,
    buffer_max_size=50,
    buffer_flush_interval=0.025,
) as db:
    # Individual calls return immediately (<10ms)
    db.insert(user1)
    db.update("u_001", user1_updated)

    # Writes are seamlessly batched and flushed in the background
```

### High-Speed Batch Updates (`update_batch`)

To update multiple entities in a single round-trip:
```python
updated_users = [user1_mod, user2_mod, user3_mod]
db.update_batch(updated_users)

# Async support
await async_db.update_batch(updated_users)
```

### Non-Blocking Forensic Audit Dispatcher

When using `ForensicModel`, WMilvus automatically offloads audit trail persistence to an asynchronous background worker (`ForensicAuditDispatcher`) with in-memory caching. This removes over 150-200ms of latency per write while guaranteeing all audit entries are safely flushed on process termination (`atexit`), `close()`, or `flush()`.

### Zero-Latency Updates via `data_before` & In-Memory LRU Cache

To eliminate the ~50-70ms read query normally needed to capture previous state for forensic auditing (`ForensicModel`):
- **Explicit Injection**: If your microservice already has the previous record in memory, pass it directly:
  ```python
  db.update(user_id, updated_user, data_before=previous_user)
  ```
- **In-Memory LRU Cache**: Entities recently inserted or queried via `get_by_field()` are automatically cached in an internal LRU cache. Subsequent updates on those IDs automatically resolve `data_before` in microsecond memory access, skipping network read calls to Milvus entirely.

### Read-After-Write Consistency & Manual Flushing

- **Immediate Visibility / Read-After-Write Consistency**: If your application or tests require immediate visibility of newly inserted entities without incurring I/O penalties, use Milvus's native `consistency_level="Strong"` or `"Session"`:
  ```python
  db = WMilvus(UserFace, uri="http://localhost:19530", consistency_level="Strong")
  ```
- **Manual Administrative Flush**: When necessary (e.g., before stopping an instance or following an offline migration), trigger a flush explicitly:
  ```python
  # Flush all registered collections
  db.flush()

  # Flush a single collection
  db[UserFace].flush()

  # Asynchronous flush
  await async_db.flush()
  ```
- **Per-operation Auto-Flush**: If you explicitly want a specific write operation to trigger a flush:
  ```python
  db.insert(user, auto_flush=True)
  db.insert_batch(users, auto_flush=True)
  db.update("u_001", user, auto_flush=True)
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
- [`examples/24_scalar_models/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/24_scalar_models/example.py) — Purely scalar / metadata models without vector fields operating alongside vector collections.
- [`examples/25_high_performance_writes/example.py`](file:///home/william.rodriguez/Documents/w_libraries/w_libraries/wmilvus_os/wmilvus/examples/25_high_performance_writes/example.py) — High-throughput writes via transparent `WriteBuffer` micro-batching and `update_batch`.

## Running Tests

Execute unit tests inside an isolated Docker container:

```bash
./run_tests_docker.sh
```

Calculate code coverage locally:

```bash
./run_coverage.sh
```
