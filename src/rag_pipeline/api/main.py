import logging
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the schemas
from rag_pipeline.api.schemas import (
    HealthCheckResponse,
    QueryRequest,
    QueryResponse,
    SourceDocumentInfo,
)
from rag_pipeline.evaluation.harness import EvaluationHarness
from rag_pipeline.generation.pipeline import GenerationPipeline
from rag_pipeline.ingestion.pipeline import IngestionPipeline

# Import the modular pipelines
from rag_pipeline.retrieval.pipeline import RetrievalPipeline

# 1. Configure a professional structured Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_api")

# 2. Initialize FastAPI Application
app = FastAPI(
    title="Multilingual Indic RAG API",
    description="Production-grade cross-lingual RAG pipeline serving English manuals to Hindi/Hinglish queries.",
    version="0.1.0",
)

# 3. Instantiate the core pipeline models at server startup
logger.info("Initializing RAG core pipelines...")
retrieval_pipeline = RetrievalPipeline()
generation_pipeline = GenerationPipeline()
ingestion_pipeline = IngestionPipeline()
evaluation_harness = EvaluationHarness()
logger.info("RAG core pipelines successfully initialized.")


# 4. Global Exception Handler: Safety net to prevent raw trace leaks
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full error on the server side for developers to debug
    logger.error(
        f"Global crash intercepted on path {request.url.path}: {str(exc)}",
        exc_info=True,
    )

    # Return a clean, standardized, secure JSON response to the client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected server-side error occurred. Please try again later."
        },
    )


# 5. Define Ingest request/response schemas directly for routing
class IngestRequest(BaseModel):
    directory_path: str = Field(
        ..., description="Absolute file path to the directory containing raw PDFs"
    )


class IngestResponse(BaseModel):
    status: str = Field(..., description="Status message of the ingestion run")
    total_chunks: int = Field(
        ..., description="Total number of chunks successfully embedded and indexed"
    )


# 6. API Route: GET /api/v1/health
@app.get(
    "/api/v1/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK
)
async def health_check() -> Dict[str, Any]:
    """
    Checks the health of the API server and verifies connectivity to the Qdrant database.
    """
    qdrant_connected = False
    try:
        # Verify connection by attempting to read collection names from Qdrant
        await retrieval_pipeline.store.client.get_collections()
        qdrant_connected = True
    except Exception as e:
        logger.warning(f"Health check failed to communicate with Qdrant: {str(e)}")

    return {
        "status": "healthy" if qdrant_connected else "degraded",
        "qdrant_connected": qdrant_connected,
        "version": "0.1.0",
    }


# 7. API Route: POST /api/v1/ingest
@app.post(
    "/api/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED
)
async def ingest_directory(payload: IngestRequest) -> Dict[str, Any]:
    """
    Asynchronously scans a target directory for PDFs, chunks them, calculates vectors,
    and indexes them inside the Qdrant database.
    """
    dir_path = Path(payload.directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provided directory path does not exist or is not a directory: {payload.directory_path}",
        )

    logger.info(f"Triggering asynchronous ingestion for directory: {dir_path}")

    # Run the full ingestion pipeline
    chunks = await ingestion_pipeline.ingest_directory(dir_path)

    if not chunks:
        return {
            "status": "No PDF files found or processed in the target directory.",
            "total_chunks": 0,
        }

    # Embed and store the chunks
    await retrieval_pipeline.store.create_collection(
        collection_name=retrieval_pipeline.collection_name,
        vector_size=retrieval_pipeline.model.dimension,
    )
    # I can call the embed pipeline directly or via store
    from rag_pipeline.embeddings.pipeline import EmbeddingPipeline

    embed_pipeline = EmbeddingPipeline(
        collection_name=retrieval_pipeline.collection_name
    )
    await embed_pipeline.embed_and_store(chunks)

    logger.info(f"Ingestion successful. Indexed {len(chunks)} chunks.")
    return {"status": "success", "total_chunks": len(chunks)}


# 8. API Route: POST /api/v1/query
@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_pipeline(payload: QueryRequest) -> Dict[str, Any]:
    """
    Receives a user query (Hindi/Hinglish/English), retrieves relevant English context chunks,
    generates a grounded Hinglish/Hindi response, and returns the answer with source references.
    """
    logger.info(f"Received query request: '{payload.query}' (top_k={payload.top_k})")

    # 1. Retrieve matching English chunks from Qdrant
    chunks = await retrieval_pipeline.retrieve(query=payload.query, top_k=payload.top_k)

    if not chunks:
        # If nothing matched, it bypass the LLM and return a clean fallback
        return {"query": payload.query, "answer": "मुझे नहीं पता।", "sources": []}

    # 2. Generate grounded Hinglish answer
    answer = await generation_pipeline.generate_answer(
        query=payload.query, context_chunks=chunks
    )

    # 3. Format source attribution data contract
    sources = [
        SourceDocumentInfo(
            source_file=chunk.metadata.source_file,
            page_number=chunk.metadata.page_number,
        )
        for chunk in chunks
    ]

    return {"query": payload.query, "answer": answer, "sources": sources}
