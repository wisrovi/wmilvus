"""Example showing ForensicModel tracking in WMilvus."""

from typing import List
from wmilvus import FieldVector, ForensicModel, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class UserFace(ForensicModel):
    """User face model with forensic audit enabled."""

    id: str
    name: str
    face_vec: List[float] = FieldVector(dim=128)


def main() -> None:
    print("--- Forensic Model Example in WMilvus ---")

    with WMilvus(UserFace, milvus_config) as db:
        user = UserFace(id="u_001", name="William Rodriguez", face_vec=[0.1] * 128)
        db.insert(user, user_id=100)
        print("Inserted UserFace with User ID=100 forensic tracking")


if __name__ == "__main__":
    main()
