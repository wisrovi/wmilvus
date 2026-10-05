"""Example showing Enterprise Ghost Table Audit Trail (_forensic_audit_log) in WMilvus.

This example demonstrates:
1. Automatic creation of global _forensic_audit_log collection when forensic=True.
2. Automatic change tracking across INSERT, UPDATE, and DELETE.
3. Querying the audit trail using db.get_ghost_audit_log() without raw SQL.
"""

from typing import List
from wmilvus import FieldVector, ForensicModel, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class BankAccountVector(ForensicModel):
    """Bank account model with forensic audit enabled."""

    id: str
    holder_name: str
    balance: float
    account_vec: List[float] = FieldVector(dim=128)


def inspect_ghost_audit_log(db: WMilvus) -> None:
    """Fetch and print recorded entries from global _forensic_audit_log ghost collection."""
    rows = db.get_ghost_audit_log()
    print(f"\n--- Global Audit Log (_forensic_audit_log) [{len(rows)} entries] ---")
    for row in rows:
        print(
            f"Audit #{row['id']} | Action: {row['action_type']} on [{row['table_name']}] "
            f"ID={row['record_id']} by User={row['create_by']}"
        )
        if row.get("data_before"):
            print(f"   BEFORE: {row['data_before']}")
        if row.get("data_after"):
            print(f"   AFTER : {row['data_after']}")
        print("-" * 60)


def main() -> None:
    print("--- Enterprise Ghost Collection Audit Trail Demonstration in WMilvus ---")

    # 1. Initialize WMilvus with a ForensicModel
    db = WMilvus(BankAccountVector, milvus_config)

    # 2. Perform INSERT with User ID=100
    print("\n1. Inserting Bank Account (User ID = 100)...")
    account = BankAccountVector(id="1", holder_name="William Rodriguez", balance=15000.00, account_vec=[0.1] * 128)
    db.insert(account, user_id=100)

    # 3. Perform UPDATE with User ID=200
    print("\n2. Updating Balance (User ID = 200)...")
    updated_account = BankAccountVector(id="1", holder_name="William Rodriguez", balance=18500.50, account_vec=[0.1] * 128)
    db.update("1", updated_account, user_id=200)

    # 4. Perform DELETE with User ID=999
    print("\n3. Deleting Account (User ID = 999)...")
    db.delete("1", user_id=999)

    # 5. Inspect recorded audit trail via ORM API
    inspect_ghost_audit_log(db)


if __name__ == "__main__":
    main()
