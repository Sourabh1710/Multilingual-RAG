# The overall flow of Pipeline:
1. Document Ingestion (loader.py & chunker.py):-
 - The 'loader.py' is used to Read files, normalizes Devanagiri/Latin characters to avoid encoding mismatches.
 - The 'chunker.py' splits long documents into smaller overlapping chunks.
 - Precautions: If the chunk size is too small then the model will fail to get the context and if the chunk size is too
                large, search results can get noisy and LLM tokens will exhaust rapidly.

2. Vector Store (Qdrant):-
 - It helps in indexing and storing vector embeddings along with document metadata(source, page, language). Performs
   High-speed vector distance calculations.
 - Precaution: I have to be careful about mismatched distance metrics (like using euclidean distance on cosine-trained
               embeddings) will result in poor search relevance.

3. Retrieval (retrieval/pipeline.py):-
 - It converts user queries into vectors and search for matching context in Qdrant.
 - Precaution: I have to avoid cross-lingual alignment gap. The query model must be perfectly align with document model,
               otherwise a Hindi query will fail to retrieve from english document.

4. Generation (generation/pipeline.py):
 - It is responsible for prompt construction, also enforces the multilingual response rules and manages the API calls to
   LLMs.
 - Precaution: Avoid prompt injection (if model fails to follow constraints) eg., responded in English instead of Hindi/
               Hinglish when context is English.

5. Evaluation (evaluation/harness.py):-
 - It measures pipeline performance scientifically using LLM-as-a-judge (Ragas).
 - Precaution: Evaluation bias or evaluation model rate-limiting due to excessive concurrent API-calling.

# Explaination of Design Decisions:
1. Why Qdrant over Pinecone? :-
    Pinecone is fully managed but closed-source and expensive while Qdrant is open-source, highly performant, written in Rust, has excellent Python async support, can be easily run locally via docker.
2. Why Asynchronous programming instead of standard synchronous code? :-
    Since RAG Pipelines are heavily I/O bound (waiting on file system reads, vector database queries,external LLM API calls), async allows us to handle thousands concurrent operations on a single thread with minimum CPU overhead.
3. Why FastAPI instead of Flask? :-
    Flask is simple but lacks native async support and auto-documentation while FastAPI built natively on async, automatically generate OpenAPI documentation, and uses Pydantic for lighting fast request/response validation.
