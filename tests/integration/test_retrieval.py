import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from rag_pipeline.embeddings.pipeline import EmbeddingPipeline
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.ingestion.loader import DocumentMetadata
from rag_pipeline.retrieval.pipeline import RetrievalPipeline


@pytest.mark.asyncio
async def test_cross_lingual_retrieval() -> None:
    # Use a unique temporary collection for this test
    test_collection = "test_retrieval_pipeline"

    # 1. Initialize the pipelines
    embed_pipeline = EmbeddingPipeline(collection_name=test_collection)
    retrieval_pipeline = RetrievalPipeline(collection_name=test_collection)

    # 2. Build a mock English document chunk
    metadata = DocumentMetadata(
        source_file="hr_leave_policy.pdf", page_number=3, total_pages=10
    )

    # An English sentence about carry forward limits
    english_content = (
        "Under the standard company benefits scheme, employees are permitted to "
        "carry forward a maximum of 15 unused casual leaves into the next calendar year. "
        "Any additional remaining leaves beyond this limit will automatically expire."
    )

    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "hr_leave_policy_p3_c0"))
    english_chunk = Chunk(
        id=chunk_id, content=english_content, chunk_index=0, metadata=metadata
    )

    # 3. Ingest and embed the English chunk into the Qdrant DB
    # embed_and_store automatically handles collection creation
    await embed_pipeline.embed_and_store([english_chunk])

    # 4. Query the database in HINGLISH (code-mixed Hindi/English)
    hinglish_query = "Casual leaves carry forward karne ki limit kya hai?"

    # 5. Run the query through the retrieval pipeline
    results = await retrieval_pipeline.retrieve(query=hinglish_query, top_k=1)

    # 6. Assertions: I expect Qdrant to find the English chunk using the Hinglish query!
    assert len(results) == 1, "The retrieval pipeline failed to return any matches!"

    matched_chunk = results[0]

    # Verify that the matched chunk is indeed the English policy document
    assert matched_chunk.metadata.source_file == "hr_leave_policy.pdf"
    assert (
        "15 unused casual leaves" in matched_chunk.content
    ), "Retrieved the wrong chunk!"

    print()
    print("SUCCESS: CROSS-LINGUAL RETRIEVAL VERIFIED!")
    print(f"Query (Hinglish): '{hinglish_query}'")
    print(f"Retrieved Context (English): '{matched_chunk.content}'")
    print()

    # Cleanup: Delete the temporary collection to leave the Docker container clean
    client = AsyncQdrantClient(host="localhost", port=6333)
    await client.delete_collection(test_collection)
