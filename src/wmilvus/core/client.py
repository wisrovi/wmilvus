"""Simplified and resilient Milvus client wrapper for vector operations."""

from typing import Any, Dict, List, Optional, Union
from loguru import logger
import numpy as np
from pymilvus import DataType, MilvusClient

from wmilvus.exceptions import ConnectionError, VectorSearchError
from wmilvus.types import SearchMatch, VectorRecord


class WMilvus:
    """High-level Milvus client wrapper designed for simplified pipeline usage."""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
        timeout: Optional[float] = 30.0,
    ) -> None:
        """Initializes the underlying MilvusClient connection.

        Args:
            uri: Target Milvus endpoint or local database path.
            token: Authentication token or API key for managed instances.
            db_name: Target logical database name.
            timeout: Network request timeout in seconds.
        """
        self.uri = uri
        try:
            self.client = MilvusClient(
                uri=uri,
                token=token,
                db_name=db_name,
                timeout=timeout,
            )
            logger.info(f"Successfully connected MilvusClient to {uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MilvusClient connection to {uri}: {e}")
            raise ConnectionError(f"Connection failed to {uri}: {e}") from e

    def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "COSINE",
        auto_id: bool = False,
    ) -> None:
        """Ensures that a vector collection and default index exist.

        Args:
            collection_name: Target collection identifier.
            dimension: Fixed length of the vector embeddings.
            metric_type: Similarity metric ('COSINE', 'L2', or 'IP').
            auto_id: Whether Milvus should generate primary keys automatically.
        """
        if not self.client.has_collection(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                auto_id=auto_id,
                id_type=DataType.VARCHAR,
                max_length=64,
            )
            logger.info(f"Created collection '{collection_name}' (dim={dimension}, metric={metric_type})")

    def upsert_batch(
        self,
        collection_name: str,
        records: List[VectorRecord],
    ) -> Dict[str, Any]:
        """Performs batch insertion or update of vector records.

        Args:
            collection_name: Target collection name.
            records: List of typed VectorRecord items.

        Returns:
            Dictionary containing operation status and modified counts.
        """
        if not records:
            return {"upsert_count": 0}

        payload = [
            {"id": item.id, "vector": item.vector, **item.metadata}
            for item in records
        ]

        result = self.client.upsert(
            collection_name=collection_name,
            data=payload,
        )
        return dict(result) if isinstance(result, dict) else {"result": result}

    def search_similar(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> List[SearchMatch]:
        """Queries the vector index for nearest neighbors.

        Args:
            collection_name: Target collection name.
            query_vector: Input embedding vector as a list or NumPy array.
            top_k: Maximum number of closest matches to retrieve.
            filter_expr: Optional scalar boolean filtering expression.
            output_fields: Specific metadata fields to return (defaults to all).

        Returns:
            List of structured SearchMatch instances sorted by similarity.
        """
        vector_data: List[float] = (
            query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        )

        try:
            search_output = self.client.search(
                collection_name=collection_name,
                data=[vector_data],
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields or ["*"],
            )

            matches: List[SearchMatch] = []
            if search_output and len(search_output) > 0:
                for item in search_output[0]:
                    entity = item.get("entity", {})
                    entity_id = str(item.get("id", entity.get("id", "")))
                    distance = float(item.get("distance", 0.0))
                    metadata = {k: v for k, v in entity.items() if k not in ("id", "vector")}
                    matches.append(SearchMatch(id=entity_id, distance=distance, metadata=metadata))

            return matches
        except Exception as e:
            logger.error(f"Vector search failed on collection '{collection_name}': {e}")
            raise VectorSearchError(f"Search failed: {e}") from e

    def delete_by_ids(self, collection_name: str, ids: List[str]) -> Dict[str, Any]:
        """Deletes vector entities matching the provided IDs.

        Args:
            collection_name: Target collection identifier.
            ids: List of primary key strings to remove.

        Returns:
            Dictionary containing deletion summary.
        """
        result = self.client.delete(
            collection_name=collection_name,
            ids=ids,
        )
        return dict(result) if isinstance(result, dict) else {"result": result}

    def drop_collection(self, collection_name: str) -> None:
        """Drops a collection and deletes all associated data and indexes.

        Args:
            collection_name: Target collection to drop.
        """
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
            logger.info(f"Dropped collection '{collection_name}'")
