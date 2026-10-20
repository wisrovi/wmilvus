# WMilvus Examples

This directory contains standalone, production-ready examples demonstrating all core features of `WMilvus`:

## Technical Stack

| Component | Technology |
|-----------|------------|
| Vector Database | Milvus 2.3+ |
| Vector ORM | WMilvus (>=0.4.0) |
| Model Validation | Pydantic v2 |
| Numerical Operations | NumPy |
| Runtime | Python 3.9+ |

## Catalog of Examples

- `01_crud/`: Single collection Pydantic ORM CRUD operations.
- `02_vector_search/`: KNN vector similarity search with distance scores and scalar metadata filtering.
- `03_range_search/`: Similarity range threshold search (radius filtering).
- `04_batch_ingestion/`: Bulk vector ingestion with automatic payload chunking (`insert_batch`).
- `05_async_client/`: Async non-blocking client (`AsyncWMilvus`) for asyncio and FastAPI.
- `06_schema_verification/`: Active Milvus collection schema verification against Pydantic models.
- `07_hybrid_search/`: Dense vector + scalar hybrid similarity search.
- `08_sqlite_backup/`: SQLite vector backup export and import powered by `wsqlite`.
- `16_forensic_fields/`: Automatic injection of forensic metadata (`create_by`, `create_in`, etc.).
- `17_multi_table/`: Managing multiple vector collections from a single `WMilvus` instance.
- `18_ghost_table_audit/`: Enterprise Ghost Table Audit Trail (`_forensic_audit_log`).
- `19_auto_id_and_int64_pk/`: Auto-increment `INT64` primary key handling and model back-population.
- `20_query_builder_filter_by/`: Fluent query builder with comparison operators (`__gte`, `__in`, `__lt`) and chained pagination.
- `21_streaming_iter_all/`: Memory-efficient streaming iterators (`iter_all`) across large collections.
- `22_partition_routing/`: Creating collection partitions, targeted partition ingestion, and partition-scoped search.
- `23_batch_and_expr_deletion/`: High-performance batch ID deletion and conditional deletion via Milvus boolean expressions.
- `24_scalar_models/`: Purely scalar / metadata models without vector fields operating alongside vector collections.
