---
name: wmilvus-architecture
description: "Architecture guide for wmilvus client, collections, vector annotations, forensic models, and wsqlite backup."
---

# `wmilvus` Architecture & Design Patterns Guide

This skill provides architectural guidance for working with or extending the `wmilvus` vector database library.

## Core Concepts

1. **Pydantic Model Vector Annotations**:
   - `FieldVector(dim: int, metric_type: MetricType, index_type: IndexType)` annotates `List[float]` fields.
   - Automatically constructs Milvus `CollectionSchema` and vector index params on collection initialization.

2. **Client Abstractions (`WMilvus` & `AsyncWMilvus`)**:
   - Manages connection lifecycle via context manager protocol (`with` and `async with`).
   - Supports single model registration (`WMilvus(UserFace)`) or multi-collection dictionary routing (`db[UserFace]`).
   - Exposes high-level CRUD (`insert`, `insert_batch`, `query`, `delete`, `upsert`) and vector search (`search_similar`, `search_similar_with_scores`, `search_range`, `search_hybrid`).

3. **SQLite Backup & Restore (`wsqlite`)**:
   - `export_to_sqlite(db, model, sqlite_path)`: Exports vector records to SQLite using `wsqlite` ORM and `WMilvusBackupRecord`.
   - `import_from_sqlite(db, model, sqlite_path)`: Restores records from SQLite backup directly back into Milvus.

4. **Forensic Audit Log (`ForensicModel` & `_forensic_audit_log`)**:
   - Models inheriting from `ForensicModel` automatically maintain audit metadata fields (`create_by`, `create_in`, `update_by`).
   - Mutations produce structured audit records in a dedicated `_forensic_audit_log` Milvus collection.
