# Tutorials

This page contains detailed, step-by-step tutorials for key features of `wmilvus`.

---

## Tutorial 1: SQLite Vector Backup & Import (`wsqlite`)

Export your Milvus collection to a local SQLite database for backup or offline analysis, and restore it later.

```python
from pydantic import BaseModel
from typing import List
from wmilvus import FieldVector, MetricType, WMilvus
from wmilvus.integrations.wsqlite import export_to_sqlite, import_from_sqlite

class UserEmbedding(BaseModel):
    id: str
    name: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)

db = WMilvus(UserEmbedding, uri="http://localhost:19530")

# 1. Populate Milvus collection
db.insert(UserEmbedding(id="1", name="Alice", vector=[0.1, 0.2, 0.3, 0.4]))
db.insert(UserEmbedding(id="2", name="Bob", vector=[0.5, 0.6, 0.7, 0.8]))

# 2. Export vector collection to SQLite backup file
backup_file = "backup_embeddings.db"
record_count = export_to_sqlite(db, UserEmbedding, backup_file)
print(f"Exported {record_count} records to {backup_file}")

# 3. Restore vector collection from SQLite backup file
restored_count = import_from_sqlite(db, UserEmbedding, backup_file)
print(f"Restored {restored_count} records to Milvus")

db.close()
```

---

## Tutorial 2: Range Search & Threshold Radius

Filter vector similarity search results by distance radius:

```python
from wmilvus import WMilvus

with WMilvus(UserEmbedding, uri="http://localhost:19530") as db:
    query_vector = [0.1, 0.2, 0.3, 0.4]

    # Radius search: find vectors within similarity distance 0.85
    results = db.search_range(vector=query_vector, radius=0.85, limit=10)
    print(f"Found {len(results)} items within distance threshold.")
```

---

## Tutorial 3: Global Enterprise Audit Log (`ForensicModel`)

Track creation and update audit trails automatically with `ForensicModel`:

```python
from wmilvus import ForensicModel, FieldVector, MetricType, WMilvus
from typing import List

class SecureAsset(ForensicModel):
    id: str
    asset_name: str
    vector: List[float] = FieldVector(dim=4, metric_type=MetricType.COSINE)

with WMilvus(SecureAsset, uri="http://localhost:19530") as db:
    asset = SecureAsset(
        id="ast_01",
        asset_name="Confidential Blueprint",
        vector=[0.1, 0.2, 0.3, 0.4],
        create_by="admin@company.com",
        create_in="System A"
    )
    db.insert(asset)
```
