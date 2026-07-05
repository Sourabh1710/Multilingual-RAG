# ADR 001: Use Qdrant as a Local, Rust-Based Vector Database

## Status
Accepted

## Context
My Retrieval-Augmented Generation (RAG) pipeline requires a high-performance vector store to index and query 384-dimensional text embeddings generated from the corporate leave manuals. The vector database must support asynchronous connections, metadata filtering, and be lightweight enough to run locally during development while remaining production-scalable.

I considered two primary options:
1.  **Pinecone:** A fully managed, cloud-native vector database.
2.  **Qdrant:** An open-source, high-speed vector database written in Rust with native gRPC/REST transport layers.

## Decision
I chose **Qdrant** as the core vector database. I package and run Qdrant locally using Docker and orchestrate it alongside the web server using Docker Compose.

## Consequences

### Consequences (Pros)
- **Local Isolation & Security:** No company data leaves the local network during development. This is highly preferred for secure corporate policies compared to closed-source cloud databases.
- **Outstanding Performance:** Being written in Rust, Qdrant is incredibly fast and memory-safe.
- **Excellent Async Support:** The `qdrant-client` library integrates natively with Python's `asyncio` event loop, preventing I/O blocking during database transactions.
- **Modern Query API:** Supports the unified `query_points` endpoint, simplifying our code and combining search, filtering, and payload retrieval into a single database handshake.

### Consequences (Cons)
- **Orchestration Overhead:** Requires setting up and maintaining a local Docker container and volume mapping to persist data between database restarts.
- **Strict ID Constraint:** Qdrant strictly enforces that point IDs must be valid unsigned integers or RFC 4122 UUID strings, requiring me to implement a deterministic hashing mechanism (`uuid.uuid5`) to map the raw string chunk IDs.
