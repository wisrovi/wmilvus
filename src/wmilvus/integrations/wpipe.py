"""WMilvus integration step for WPipe pipeline execution."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from wmilvus.core.client import WMilvus


class WMilvusIngestStep:
    """Pipeline step for ingesting vector records into Milvus within WPipe data workflows."""

    def __init__(
        self,
        model_cls: Type[BaseModel],
        uri: str = "http://localhost:19530",
        batch_size: int = 1000,
    ) -> None:
        self.model_cls = model_cls
        self.uri = uri
        self.batch_size = batch_size

    def process(self, records: List[BaseModel]) -> Dict[str, Any]:
        """Ingest records batch into Milvus DB collection.

        Args:
            records: List of typed Pydantic models.

        Returns:
            Dict summary containing inserted_count and status.
        """
        if not records:
            return {"inserted_count": 0, "status": "SKIPPED"}

        with WMilvus(self.model_cls, uri=self.uri) as db:
            result = db.insert_batch(records, batch_size=self.batch_size)
            return {
                "inserted_count": result["inserted_count"],
                "batches": result["batches_processed"],
                "status": "SUCCESS",
            }
