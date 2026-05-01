# ADR-001: SQLite, FTS5, and sqlite-vec

Lexa uses SQLite as the single local store. FTS5 handles lexical retrieval, and sqlite-vec stores float and binary vector tables. This keeps indexing, deletion, and backup behavior simple.
