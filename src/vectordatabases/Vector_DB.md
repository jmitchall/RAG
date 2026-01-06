# Vector DB choices — Qdrant, Faiss, Chroma

When building a RAG pipeline you must pick a vector store. Below are concise pros/cons to help decide which to use with your RTX 5080 16GB local stack.

### At a glance
- Qdrant — managed-like, feature-rich, persistent, easy remote/hosted use.
- Faiss — fastest, memory-efficient, GPU-accelerated (but low-level).
- Chroma — developer-friendly, integrated with LangChain, good defaults.

### Qdrant
Pros:
- Persistent storage with a production-ready HTTP/gRPC server (easy to run locally or remotely).
- Built-in metadata filtering, payload support, and advanced search features.
- Horizontal scaling and snapshot/backup support.
- Vector indexes can be stored on-disk — good for larger datasets.
- SDKs for Python and other languages; integrates well with modern apps.

Cons:
- Requires running a separate service (container or binary) — more operational overhead than in-process stores.
- Slightly higher latency vs in-process libraries (Faiss) for small datasets.
- Memory and disk usage depend on chosen index type and settings.

Best when: you want a production-grade vector DB with metadata filtering, persistence, and remote access.

### Faiss
Pros:
- Extremely fast and memory-efficient nearest neighbor search (CPU + GPU support).
- Mature library from Facebook Research with many index types (IVF, HNSW, PQ, OPQ, etc.).
- Best raw performance for large-scale similarity search when tuned correctly.
- Can be embedded in-process (no server) for minimal latency.

Cons:
- Low-level API — more engineering effort to manage indexes, persistence, and metadata.
- Persistence requires manual save/load of index files and separate metadata store.
- GPU usage needs correct CUDA setup and careful memory management.
- Less out‑of‑the‑box functionality for filtering and metadata compared to Qdrant/Chroma.

Best when: you need maximum performance and control, and can manage index persistence and metadata yourself.

### Chroma
Pros:
- Easy to use; designed for embeddings + simple metadata; great developer UX.
- Integrates tightly with LangChain and common embedding workflows.
- Provides on-disk persistence and optional in-memory backends.
- Good defaults for small-to-medium projects and prototypes.

Cons:
- Slower/scaling limitations vs Faiss for very large corpora.
- Feature set and performance depend on the chosen backend (in-process vs SQLite/file).
- Not as feature-complete for production metadata/replication/scaling as Qdrant.

Best when: you want fast development, easy LangChain integration, and on-disk persistence without operating a separate DB service.