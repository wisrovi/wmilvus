---
name: milvus-search-guide
description: "Best practices and code patterns for vector similarity KNN search, range search, distance metrics, and filtering in wmilvus."
---

# Milvus Vector Search Guide for `wmilvus`

This guide explains how to query vectors efficiently using `wmilvus`.

## Search Operations

1. **Similarity KNN Search (`search_similar`)**:
   - Performs top-k nearest neighbor vector similarity search.
   - Returns typed Pydantic instances directly.
   ```python
   results = db.search_similar(vector=[0.1] * 128, top_k=5, filter_expr='category == "A"')
   ```

2. **Similarity with Distance Scores (`search_similar_with_scores`)**:
   - Returns results alongside distance metric scores (`MetricType.COSINE`, `MetricType.L2`, `MetricType.IP`).
   ```python
   matches = db.search_similar_with_scores(vector=[0.1] * 128, top_k=5)
   for match in matches:
       print(match.item.id, match.distance)
   ```

3. **Radius Range Search (`search_range`)**:
   - Filters matches by distance threshold boundaries.
   ```python
   near_items = db.search_range(vector=[0.1] * 128, radius=0.85, limit=10)
   ```

4. **Hybrid Search (`search_hybrid`)**:
   - Combines vector embeddings with scalar metadata filter expressions (e.g. status, tags, timestamps).
