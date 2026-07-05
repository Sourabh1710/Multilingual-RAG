# ADR 003: Choose Asynchronous asyncio and Cooperative Multitasking for High-Throughput I/O

## Status
Accepted

## Context
My ingestion pipeline is heavily I/O-bound. It must scan directories, read binary PDF files from disk, normalize Unicode text, send data batches across the network to generate embeddings, and upload vectors to my Qdrant database.

In a traditional synchronous python application, the execution thread spends 95% of its time idling and waiting on the filesystem or network sockets, severely limiting ingestion throughput.

I've considered:
1.  **Multi-threading:** Uses OS threads, but is heavily restricted by Python's Global Interpreter Lock (GIL) and consumes high CPU memory overhead.
2.  **Asynchronous Cooperative Multitasking (asyncio):** A single-threaded event loop that yields control (`await`) during I/O waits to process other tasks concurrently.

## Decision
I chose **`asyncio`** as the core runtime for our entire pipeline and API.

## Consequences

### Consequences (Pros)
- **Incredible Efficiency:** I can handle hundreds of concurrent file reads and network requests on a single thread with virtually zero CPU overhead.
- **Native FastAPI Integration:** FastAPI is built natively on ASGI and asyncio, making our pipelines, database stores, and API routers completely compatible without requiring thread-safety wrappers.
- **Safer Resource Limits:** I've configure an `asyncio.Semaphore` inside the ingestion pipeline to restrict concurrent file reads, ensuring it never exceed operating system file descriptor limits (preventing "Too many open files" crashes).

### Consequences (Cons)
- **The Starvation Risk:** Since asyncio is single-threaded, any heavy CPU-bound calculation (like running the `SentenceTransformer` matrix multiplications) will block the Event Loop and freeze the entire server. I've mitigate this by explicitly offloading our embedding generation (`encode`) to background worker threads using `asyncio.to_thread()`.
