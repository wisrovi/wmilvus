"""Example demonstrating Collection Partition Routing and Scoped Search in WMilvus."""

from typing import List
from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class DocumentChunk(BaseModel):
    """Pydantic model representing chunked document vectors partitioned by tenant/region."""

    __tablename__ = "RegionalDocumentChunks"

    id: str = Field(..., description="Document chunk UUID")
    tenant_region: str = Field(..., description="Geographical tenant region")
    title: str = Field(..., description="Document title")
    embedding: List[float] = FieldVector(dim=128, metric_type=MetricType.COSINE)


def main() -> None:
    print("--- Partition Routing and Scoped Search in WMilvus ---")

    with WMilvus(DocumentChunk, milvus_config) as db:
        # 1. Create regional partitions in the Milvus collection
        db.create_partition("region_europe")
        db.create_partition("region_americas")
        print("Created partitions: region_europe, region_americas")

        # 2. Insert records targeted into specific partitions
        doc_eu = DocumentChunk(
            id="doc_eu_01",
            tenant_region="Europe",
            title="EU Compliance Guidelines 2026",
            embedding=[0.08] * 128,
        )
        doc_us = DocumentChunk(
            id="doc_us_01",
            tenant_region="Americas",
            title="US Market Dynamics 2026",
            embedding=[0.08] * 128,
        )

        db.insert(doc_eu, partition_name="region_europe")
        db.insert(doc_us, partition_name="region_americas")
        print("Inserted documents into their respective partitions.")

        # 3. Vector search strictly scoped to the 'region_europe' partition
        eu_matches = db.search_similar(
            vector=[0.08] * 128,
            partition_names=["region_europe"],
            top_k=5,
        )
        print(f"\nEurope partition matches ({len(eu_matches)}):")
        for match in eu_matches:
            print(f" - {match.id}: {match.title} [{match.tenant_region}]")

        # 4. Vector search strictly scoped to the 'region_americas' partition
        us_matches = db.search_similar(
            vector=[0.08] * 128,
            partition_names=["region_americas"],
            top_k=5,
        )
        print(f"\nAmericas partition matches ({len(us_matches)}):")
        for match in us_matches:
            print(f" - {match.id}: {match.title} [{match.tenant_region}]")


if __name__ == "__main__":
    main()
