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

**wmilvus** is an enterprise-grade, type-safe Milvus Vector Database client and wrapper that leverages Pydantic models for schema definitions, collection auto-creation, index configuration, and vector similarity search.

## Key Features

- **Pydantic Integration** — Map vector collection metadata and payload entities using Pydantic models.
- **Type-Safe Search** — High-level, type-safe API for vector insertion, collection management, and top-k similarity queries.
- **Auto Indexing** — Declarative configuration of vector dimensions and indexing parameters (`HNSW`, `IVF_FLAT`, `FLAT`).
- **Connection Management** — Handles lifecycle and parameters for Milvus server / cluster connections seamlessly.

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Vector Engine | Milvus 2.3+ |
| SDK Core | pymilvus 2.3+ |
| Validation | Pydantic 2.x |
| Logging | Loguru |
| CLI | Click |
| Testing | pytest, pytest-cov |
| Linting | ruff |

## Quick Start

```python
from pydantic import BaseModel
from wmilvus import WMilvus, VectorFieldConfig

class ImageVector(BaseModel):
    image_id: str
    description: str

config = VectorFieldConfig(dim=128, metric_type="COSINE", index_type="HNSW")
client = WMilvus(model_class=ImageVector, vector_config=config, host="localhost", port=19530)
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