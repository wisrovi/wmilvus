"""Example demonstrating SQLite Backup Export & Import in WMilvus."""

from typing import List
from pydantic import BaseModel
from wmilvus import FieldVector, MetricType, WMilvus, export_to_sqlite, import_from_sqlite

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class UserProfile(BaseModel):
    """User profile model for backup and restore testing."""

    id: str
    username: str
    bio: str
    vector: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- WSQLite Backup & Restore Demonstration ---")
    sqlite_backup_file = "user_profiles_backup.db"

    with WMilvus(UserProfile, milvus_config) as db:
        # 1. Insert sample profiles into Milvus
        profiles = [
            UserProfile(id="u101", username="alice", bio="AI Developer", vector=[0.1] * 128),
            UserProfile(id="u102", username="bob", bio="Data Engineer", vector=[0.2] * 128),
        ]
        db.insert_batch(profiles)
        print("Inserted 2 profiles into Milvus!")

        # 2. Export Milvus collection to SQLite database
        export_result = export_to_sqlite(db, UserProfile, sqlite_backup_file)
        print(f"Exported {export_result['exported_count']} records to '{sqlite_backup_file}'!")

        # 3. Clean up Milvus records
        db.delete("u101")
        db.delete("u102")
        print("Deleted original records from Milvus!")

        # 4. Restore records from SQLite backup back into Milvus
        import_result = import_from_sqlite(db, UserProfile, sqlite_backup_file)
        print(f"Restored {import_result['imported_count']} records from SQLite into Milvus!")

        # 5. Verify restored data
        all_profiles = db.get_all()
        print("Restored User Profiles:", [p.username for p in all_profiles])


if __name__ == "__main__":
    main()
