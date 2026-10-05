"""Backup and restore utilities for WMilvus using WSQLite ORM."""

from typing import Any, Dict, Type
from pydantic import BaseModel, Field

from wmilvus.core.client import WMilvus


class WMilvusBackupRecord(BaseModel):
    """Pydantic model representing a backed-up vector record for WSQLite storage."""

    id: str = Field(description="Unique record identifier")
    payload_json: str = Field(description="Serialized JSON representation of the Pydantic model record")


def export_to_sqlite(
    db: WMilvus,
    model_cls: Type[BaseModel],
    sqlite_path: str,
    limit: int = 10000,
) -> Dict[str, Any]:
    """Export all records from a WMilvus collection into a local SQLite database file via WSQLite ORM.

    Args:
        db: Initialized WMilvus client instance.
        model_cls: Pydantic model bound to the collection.
        sqlite_path: Target SQLite database file path.
        limit: Maximum number of records to export.

    Returns:
        Dict summary with exported_count and sqlite_path.
    """
    from wsqlite import WSQLite

    repo = db[model_cls] if model_cls in db.repositories else db
    records = repo.get_all(limit=limit)

    count = 0
    sqlite_db = WSQLite(WMilvusBackupRecord, sqlite_path)
    try:
        for rec in records:
            rec_id = str(getattr(rec, "id", ""))
            rec_json = rec.model_dump_json()
            backup_rec = WMilvusBackupRecord(id=rec_id, payload_json=rec_json)
            sqlite_db.insert(backup_rec)
            count += 1
    finally:
        if hasattr(sqlite_db, "close"):
            sqlite_db.close()

    return {"exported_count": count, "sqlite_path": sqlite_path}


def import_from_sqlite(
    db: WMilvus,
    model_cls: Type[BaseModel],
    sqlite_path: str,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Import records from a WSQLite database file back into a WMilvus collection via WSQLite ORM.

    Args:
        db: Initialized WMilvus client instance.
        model_cls: Target Pydantic model class.
        sqlite_path: Path to the WSQLite database file.
        batch_size: Ingestion batch size for Milvus.

    Returns:
        Dict summary containing imported_count and status.
    """
    from wsqlite import WSQLite

    sqlite_db = WSQLite(WMilvusBackupRecord, sqlite_path)
    try:
        backup_records = sqlite_db.get_all()
    finally:
        if hasattr(sqlite_db, "close"):
            sqlite_db.close()

    if not backup_records:
        return {"imported_count": 0, "status": "SKIPPED"}

    records = [model_cls.model_validate_json(rec.payload_json) for rec in backup_records]
    result = db.insert_batch(records, batch_size=batch_size)

    return {
        "imported_count": result["inserted_count"],
        "status": "SUCCESS",
        "sqlite_path": sqlite_path,
    }
