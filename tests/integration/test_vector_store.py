import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from rag_pipeline.embeddings.model import EmbeddingModel
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.ingestion.loader import DocumentMetadata
from rag_pipeline.retrieval.store import QdrantStore


# 1. Integration Test to verify the local model dimensionality
def test_embedding_model_dimensionality() -> None:
    # Initialize the model
    embedder = EmbeddingModel()

    # Assert that the dimensions match what I expect (MiniLM has 384 dimensions)
    assert embedder.dimension == 384, "The embedding model dimension is not 384!"

    # Encode a mock text list
    vectors = embedder._encode_sync(["Test sentence"])

    # Assert that the returned vector length matches the model specification
    assert len(vectors) == 1
    assert (
        len(vectors[0]) == 384
    ), f"Generated vector has incorrect size: {len(vectors[0])}"


# 2. Integration Test to perform a real write/read roundtrip to the running Docker Qdrant
@pytest.mark.asyncio
async def test_qdrant_roundtrip() -> None:
    # Use a temporary collection name for testing to avoid overwriting production data
    test_collection = "test_integration_chunks"

    # Initialize the store wrapper
    store = QdrantStore(host="localhost", port=6333)
    embedder = EmbeddingModel()

    # 1. Create the temporary collection in Qdrant
    await store.create_collection(
        collection_name=test_collection, vector_size=embedder.dimension
    )

    # 2. Build a mock Chunk object
    metadata = DocumentMetadata(
        source_file="integration_test.pdf", page_number=1, total_pages=1
    )
    # Generate a valid UUID for the mock chunk
    test_string_id = "integration_test_pdf_p1_c0"
    test_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, test_string_id))

    mock_chunk = Chunk(
        id=test_uuid,
        content="This is a unique string used to test the vector database integration.",
        chunk_index=0,
        metadata=metadata,
    )

    # 3. Generate the vector embedding for the chunk
    vectors = await embedder.encode([mock_chunk.content])

    # 4. Upsert the chunk and vector into Qdrant
    await store.upsert_chunks(
        collection_name=test_collection, chunks=[mock_chunk], embeddings=vectors
    )

    # 5. Let's query Qdrant directly to verify the data is physically there
    client = AsyncQdrantClient(host="localhost", port=6333)

    # Retrieve the collection info and assert points exist
    collection_info = await client.get_collection(test_collection)
    assert collection_info.points_count is not None
    assert (
        collection_info.points_count > 0
    ), "No points were successfully written to Qdrant!"

    # Cleanup: Delete the test collection after test finishes to keep the database clean
    await client.delete_collection(test_collection)
