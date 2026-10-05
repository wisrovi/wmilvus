# Frequently Asked Questions (FAQ)

## 1. How does `wmilvus` map Pydantic types to Milvus schemas?
`wmilvus` inspects model fields annotated with `FieldVector`. It creates a Milvus `CollectionSchema` with primary key (`id`), scalar fields (`str`, `int`, `float`), and vector fields (`FloatVector`).

---

## 2. Why aren't inserted records immediately visible in search queries?
In Milvus Standalone, vector insertions are held in memory before being committed to segments. `wmilvus` triggers `flush()` automatically on `insert` and `insert_batch` to guarantee immediate query availability.

---

## 3. How does SQLite backup work without raw `sqlite3` cursors?
`wmilvus` uses `wsqlite` ORM and Pydantic data models (`WMilvusBackupRecord`) to serialise vectors as JSON strings into a SQLite table, avoiding raw SQL queries or cursor code.

---

## 4. Is `AsyncWMilvus` thread-safe?
Yes, `AsyncWMilvus` delegates blocking operations to `pymilvus` via async event loop thread pools.
