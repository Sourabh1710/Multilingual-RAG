import uuid

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient
from rag_pipeline.api.main import app, retrieval_pipeline
from rag_pipeline.embeddings.pipeline import EmbeddingPipeline
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.ingestion.loader import DocumentMetadata


# 1. Test to verify the HTTP Health Check endpoint
@pytest.mark.asyncio
async def test_api_health_check() -> None:
    # Used httpx.AsyncClient as our in-memory test client, passing the FastAPI 'app'
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/api/v1/health")

        # Assertions: Verify it get a clean 200 OK status
        assert response.status_code == status.HTTP_200_OK

        response_json = response.json()
        assert response_json["status"] == "healthy"
        assert response_json["qdrant_connected"] is True


# 2. Test to verify the full Query endpoint (Retrieval + Generation over HTTP)
@pytest.mark.asyncio
async def test_api_query_endpoint() -> None:
    # Ensure the database has at least one chunk indexed for testing
    test_collection = retrieval_pipeline.collection_name
    embed_pipeline = EmbeddingPipeline(collection_name=test_collection)

    metadata = DocumentMetadata(
        source_file="api_test_manual.pdf", page_number=1, total_pages=5
    )
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "api_test_manual_p1_c0"))
    mock_chunk = Chunk(
        id=chunk_id,
        content="Employees must submit leave requests via the HR Portal at least 2 days prior to the start date.",
        chunk_index=0,
        metadata=metadata,
    )
    # Seed the Qdrant database with the mock chunk
    await embed_pipeline.embed_and_store([mock_chunk])

    # Construct the Query JSON payload following the Pydantic QueryRequest contract
    payload = {"query": "Leave request submit karne ki deadline kya hai?", "top_k": 1}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # I have send an asynchronous HTTP POST request to the query endpoint
        response = await ac.post("/api/v1/query", json=payload)

        # Assertions: Verify that it get a clean 200 OK status
        assert response.status_code == status.HTTP_200_OK

        # Assertions: Verify the response conforms exactly to the Pydantic QueryResponse contract
        response_json = response.json()
        assert "query" in response_json
        assert "answer" in response_json
        assert "sources" in response_json

        # Verify that the returned sources are formatted correctly
        sources = response_json["sources"]
        assert len(sources) == 1
        assert sources[0]["source_file"] == "api_test_manual.pdf"
        assert sources[0]["page_number"] == 1

        # Verify that the system prompt successfully grounded the answer in the seed data
        answer = response_json["answer"].lower()
        assert (
            "hr portal" in answer
            or "calendar year" in answer
            or "casual leave" in answer
        )
