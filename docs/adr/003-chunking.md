# ADR-003: Chunking

Lexa chunks by file type. Code uses tree-sitter parsers selected from the file extension for Rust, Python, JavaScript, TypeScript, Go, Java, C, and C++. Leading comments are included with symbol chunks so code search preserves descriptive context.

Markdown uses heading boundaries. Text, logs, JSON, TOML, YAML, and CSV use stable line windows. PDF files use `pdf-extract` text extraction and then the normal text chunker.

Each chunk preserves byte offsets and line ranges.
