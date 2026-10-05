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

    def close(self) -> None:
        """Close underlying client connections."""
        if hasattr(self.client, "close"):
            try:
                self.client.close()
                logger.info("Closed MilvusClient connection")
            except Exception as e:
                logger.warning(f"Error while closing MilvusClient: {e}")

    def __enter__(self) -> "WMilvus":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Performs batch insertion or update of vector records with automatic chunking.

        Args:
            collection_name: Target collection name.
            records: List of typed VectorRecord items.
            chunk_size: Maximum number of records per upsert request payload.

        Returns:
            Dictionary containing operation status and modified counts.
        """
        if not records:
            return {"upsert_count": 0}

        total_upserted = 0
        last_result: Any = None

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            payload = [
                {"id": item.id, "vector": item.vector, **item.metadata}
                for item in chunk
            ]

            res = self.client.upsert(
                collection_name=collection_name,
                data=payload,
            )
            last_result = res
            if isinstance(res, dict) and "upsert_count" in res:
                total_upserted += int(res["upsert_count"])
            else:
                total_upserted += len(chunk)

        if isinstance(last_result, dict):
            summary = dict(last_result)
            summary["upsert_count"] = total_upserted
            return summary
        return {"upsert_count": total_upserted, "result": last_result}

    def search_similar(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> List[SearchMatch]:
        """Queries the vector index for nearest neighbors for a single vector.

        Args:
            collection_name: Target collection name.
            query_vector: Input embedding vector as a list or NumPy array.
            top_k: Maximum number of closest matches to retrieve.
            filter_expr: Optional scalar boolean filtering expression.
            output_fields: Specific metadata fields to return (defaults to all).

        Returns:
            List of structured SearchMatch instances sorted by similarity.
        """
        results = self.search_batch(
            collection_name=collection_name,
            query_vectors=[query_vector],
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return results[0] if results else []

    def search_batch(
        self,
        collection_name: str,
        query_vectors: Union[List[List[float]], List[np.ndarray], np.ndarray],
        top_k: int = 5,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> List[List[SearchMatch]]:
        """Queries the vector index for nearest neighbors for a batch of query vectors.

        Args:
            collection_name: Target collection name.
            query_vectors: Batch of input embedding vectors (list of lists, list of arrays, or 2D NumPy array).
            top_k: Maximum number of closest matches to retrieve per query vector.
            filter_expr: Optional scalar boolean filtering expression.
            output_fields: Specific metadata fields to return (defaults to all).

        Returns:
            List of lists of SearchMatch instances sorted by similarity for each query vector.
        """
        if isinstance(query_vectors, np.ndarray):
            vectors_list: List[List[float]] = query_vectors.tolist()
        else:
            vectors_list = [
                v.tolist() if isinstance(v, np.ndarray) else v for v in query_vectors
            ]

        if not vectors_list:
            return []

        try:
            search_output = self.client.search(
                collection_name=collection_name,
                data=vectors_list,
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields or ["*"],
            )

            batch_matches: List[List[SearchMatch]] = []
            if search_output:
                for single_search_res in search_output:
                    matches: List[SearchMatch] = []
                    for item in single_search_res:
                        entity = item.get("entity", {})
                        entity_id = str(item.get("id", entity.get("id", "")))
                        distance = float(item.get("distance", 0.0))
                        metadata = {k: v for k, v in entity.items() if k not in ("id", "vector")}
                        matches.append(SearchMatch(id=entity_id, distance=distance, metadata=metadata))
                    batch_matches.append(matches)

            return batch_matches
        except Exception as e:
            logger.error(f"Vector search failed on collection '{collection_name}': {e}")
            raise VectorSearchError(f"Search failed: {e}") from e

    def query_scalar(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieves entities matching a scalar boolean expression without vector similarity search.

        Args:
            collection_name: Target collection identifier.
            filter_expr: Scalar boolean expression (e.g. 'camera_id == "cam_1"').
            output_fields: List of metadata field names to return.
            limit: Maximum number of entities to retrieve.

        Returns:
            List of dictionaries containing matched entity metadata.
        """
        try:
            return self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["*"],
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Scalar query failed on collection '{collection_name}': {e}")
            raise VectorSearchError(f"Query failed: {e}") from e

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
