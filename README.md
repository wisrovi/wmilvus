# wmilvus

<p align="center">
    <a href="https://pypi.org/project/wmilvus/">
        <img src="https://img.shields.io/pypi/v/wmilvus.svg" alt="PyPI version">
    </a>
    <a href="https://pypi.org/project/wmilvus/">
        <img src="https://img.shields.io/pypi/pyversions/wmilvus.svg" alt="Python versions">
    </a>
    <a href="https://github.com/wisrovi/wmilvus/blob/main/LICENSE">
        <img src="https://img.shields.io/pypi/l/wmilvus.svg" alt="License">
    </a>
</p>

**wmilvus** is an enterprise-grade, type-safe Milvus Vector Database client wrapper designed for simplified pipeline usage, batch vector operations, and metadata querying.

## Key Features

- **Pydantic Data Models** — Typed data transfer objects (`VectorRecord`, `SearchMatch`) for vector entities and metadata.
- **Batch Vector Operations** — High-performance single and batch vector similarity searches (`search_similar`, `search_batch`).
- **Resilient Batch Upserting** — Automatic payload chunking (`chunk_size=1000`) for large-scale embedding ingestion.
- **Scalar Metadata Queries** — Filter vector entities by boolean scalar expressions (`query_scalar`) without requiring vector search.
- **Context Manager Support** — Clean resource cleanup using Python `with` statements or explicit `.close()`.

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Vector Engine | Milvus 2.3+ |
| SDK Core | pymilvus 2.3+ |
| Validation | Pydantic 2.x |
| Numerical Core | NumPy |
| Logging | Loguru |
| CLI | Click |
| Testing | pytest, pytest-cov |
| Linting | ruff |

## Quick Start

```python
import numpy as np
from wmilvus import WMilvus, VectorRecord

# Initialize client using context manager
with WMilvus(uri="http://localhost:19530") as milvus:
    # 1. Ensure collection exists
    milvus.ensure_collection(collection_name="visual_embeddings", dimension=512, metric_type="COSINE")

    # 2. Upsert batch of vector records
    sample_vector = np.random.rand(512).astype(np.float32).tolist()
    records = [
        VectorRecord(
            id="frame_00104",
            vector=sample_vector,
            metadata={"camera_id": "cam_entry_north", "label": "forklift"},
        )
    ]
    milvus.upsert_batch(collection_name="visual_embeddings", records=records)

    # 3. Perform single vector similarity search
    query = np.random.rand(512).astype(np.float32)
    matches = milvus.search_similar(
        collection_name="visual_embeddings",
        query_vector=query,
        top_k=3,
        filter_expr='label == "forklift"',
    )

    # 4. Perform scalar metadata query
    items = milvus.query_scalar(
        collection_name="visual_embeddings",
        filter_expr='camera_id == "cam_entry_north"',
    )
```

## Running Tests

Execute tests in an isolated Docker container:

```bash
./run_tests_docker.sh
```

Calculate code coverage:

```bash
./run_coverage.sh
```