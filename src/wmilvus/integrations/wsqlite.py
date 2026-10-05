"""Backup and restore utilities for WMilvus using SQLite as portable storage format."""

import json
import sqlite3
from typing import Any, Dict, Type
from pydantic import BaseModel

from wmilvus.core.client import WMilvus


def export_to_sqlite(
    db: WMilvus,
    model_cls: Type[BaseModel],
    sqlite_path: str,
    table_name: str = "vector_records",
    limit: int = 10000,
) -> Dict[str, Any]:
    """Export all records from a WMilvus collection into a local SQLite database file.

    Args:
        db: Initialized WMilvus client instance.
        model_cls: Pydantic model bound to the collection.
        sqlite_path: Target SQLite database file path.
        table_name: SQLite table name to create and populate.
        limit: Maximum number of records to export.

    Returns:
        Dict summary with exported_count and sqlite_path.
    """
    repo = db[model_cls] if model_cls in db.repositories else db
    records = repo.get_all(limit=limit)

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL
        )
    """
    )

    count = 0
    for rec in records:
        rec_id = str(getattr(rec, "id", ""))
        rec_json = rec.model_dump_json()
        cursor.execute(
            f"INSERT OR REPLACE INTO {table_name} (id, payload_json) VALUES (?, ?)",
            (rec_id, rec_json),
        )
        count += 1

    conn.commit()
    conn.close()

    return {"exported_count": count, "sqlite_path": sqlite_path, "table_name": table_name}


def import_from_sqlite(
    db: WMilvus,
    model_cls: Type[BaseModel],
    sqlite_path: str,
    table_name: str = "vector_records",
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Import records from a SQLite database file back into a WMilvus collection.

    Args:
        db: Initialized WMilvus client instance.
        model_cls: Target Pydantic model class.
        sqlite_path: Path to the SQLite database file.
        table_name: SQLite table name containing exported records.
        batch_size: Ingestion batch size for Milvus.

    Returns:
        Dict summary containing imported_count and status.
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT payload_json FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {"imported_count": 0, "status": "SKIPPED"}

    records = [model_cls.model_validate_json(row[0]) for row in rows]
    result = db.insert_batch(records, batch_size=batch_size)

    return {
        "imported_count": result["inserted_count"],
        "status": "SUCCESS",
        "sqlite_path": sqlite_path,
    }
