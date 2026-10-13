"""Example demonstrating Batch and Expression Deletion in WMilvus."""

from typing import List
from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class CacheItem(BaseModel):
    """Pydantic model representing ephemeral cached embeddings."""

    __tablename__ = "VectorCacheEntries"

    id: str = Field(..., description="Unique cache item key")
    status: str = Field(..., description="Lifecycle status: active, expired, archived")
    embedding: List[float] = FieldVector(dim=64, metric_type=MetricType.IP)


def main() -> None:
    print("--- Batch and Expression Deletion in WMilvus ---")

    with WMilvus(CacheItem, milvus_config) as db:
        # 1. Ingest test items
        items = [
            CacheItem(id="cache_01", status="active", embedding=[0.1] * 64),
            CacheItem(id="cache_02", status="expired", embedding=[0.2] * 64),
            CacheItem(id="cache_03", status="expired", embedding=[0.3] * 64),
            CacheItem(id="cache_04", status="archived", embedding=[0.4] * 64),
            CacheItem(id="cache_05", status="archived", embedding=[0.5] * 64),
        ]
        db.insert_batch(items)
        print(f"Ingested {len(items)} cache items.")

        # 2. Batch deletion by explicit list of primary keys
        print("\nDeleting batch by primary keys: ['cache_02', 'cache_03']")
        db.delete_batch(["cache_02", "cache_03"])
        remaining_after_batch = db.get_all()
        print(f"Remaining IDs after delete_batch: {[item.id for item in remaining_after_batch]}")

        # 3. Conditional deletion using boolean Milvus expressions
        print("\nDeleting by expression: status == 'archived'")
        db.delete_by_expr('status == "archived"')
        final_items = db.get_all()
        print(f"Final remaining IDs in collection: {[item.id for item in final_items]}")


if __name__ == "__main__":
    main()
