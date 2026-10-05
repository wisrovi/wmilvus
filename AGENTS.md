# AGENTS.md — Agent Guidelines for `wmilvus`

Welcome AI Agent! This file defines the repository architecture, developer instructions, coding standards, and operational guidelines for working on **`wmilvus`**.

---

## 1. Repository Overview

`wmilvus` is an enterprise-grade, type-safe Python client wrapper for the **Milvus Vector Database**. It maps Pydantic models to Milvus collections, handles vector indexing, multi-collection routing, async context management, enterprise forensic audit trails (`_forensic_audit_log`), and SQLite vector backup/restoration (`wsqlite`).

### Core Technologies
- **Python**: 3.9+
- **Vector Database**: Milvus 2.3+ (pymilvus)
- **Data Schema & Validation**: Pydantic 2.x
- **Backup & Import ORM**: `wsqlite` 1.5+
- **Numerical Core**: NumPy
- **Logging**: Loguru
- **Testing**: `pytest`, `pytest-cov`, Docker (`run_tests_docker.sh`)

---

## 2. Directory Structure

```
wmilvus/
├── src/wmilvus/
│   ├── core/           # WMilvus & AsyncWMilvus client implementation
│   ├── types/          # FieldVector annotations, MetricType, IndexType, ForensicModel
│   ├── integrations/   # wpipe step & wsqlite vector backup/restore
│   ├── exceptions/     # Custom WMilvus exception hierarchy
│   └── cli/            # WMilvus CLI commands
├── tests/              # Unit & integration tests
├── examples/           # Runnable example scripts (01_crud to 18_ghost_table_audit)
├── .agents/skills/     # Local agent skills for wmilvus architecture & vector search
├── run_tests_docker.sh # Dockerized test runner script
└── run_coverage.sh     # Local test coverage report generator
```

---

## 3. Developer & Agent Rules

1. **Language Standards**:
   - Code, docstrings, inline comments, and commit messages MUST be written in **English**.
   - Type annotations must be precise and compatible with Python 3.9+.

2. **Testing & Quality Assurance**:
   - Always run unit tests via Docker: `./run_tests_docker.sh`.
   - Ensure code coverage remains high: `./run_coverage.sh`.
   - Never remove or break existing unit tests.

3. **Git Commit Rules**:
   - Strict **1 file per commit** policy.
   - Commit messages must start with a category tag in brackets (e.g. `[FEATURE]`, `[FIX]`, `[DOCS]`, `[TEST]`, `[CHORE]`).
   - Validate pre-commit hooks (`pre-commit run --files ...`) prior to committing.

4. **Documentation**:
   - Keep `README.md` synchronized whenever features or examples are added or changed.
   - Ensure `README.md` includes the Technical Stack table.

---

## 4. Agent Skills Location

Skills specific to this codebase are maintained in `.agents/skills/`:
- `.agents/skills/wmilvus-architecture/SKILL.md`: Deep dive into core client architecture and collection lifecycle.
- `.agents/skills/milvus-search-guide/SKILL.md`: Best practices for similarity KNN, range search, distance metrics, and filtering.
