import uuid

import pytest
from rag_pipeline.ingestion.chunker import RecursiveCharacterChunker
from rag_pipeline.ingestion.loader import Document, DocumentMetadata


# 1. Define a pytest fixture to create a mock document for the tests
@pytest.fixture
def mock_document() -> Document:
    metadata = DocumentMetadata(
        source_file="test_manual.pdf", page_number=1, total_pages=5
    )
    # Create a 1500-character mock text block to force chunking
    text_content = (
        "This is paragraph one of the mock manual. It contains basic setup details.\n\n"
        "This is paragraph two of the mock manual. It details the safety instructions "
        "and parameters that must be strictly followed during operational cycles.\n\n"
        "This is paragraph three. It contains troubleshooting and support information "
        "for developers integrating this RAG pipeline natively into local stacks."
    )
    return Document(content=text_content, metadata=metadata)


# 2. Write the unit tests
def test_chunk_count_and_sizing(mock_document: Document) -> None:
    # Initialize chunker with small chunk size to force multiple splits
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)

    chunks = chunker.chunk_document(mock_document)

    # Verify that it successfully generated multiple chunks
    assert len(chunks) > 1, "Chunker failed to split the document into multiple pieces"

    # Verify that no chunk exceeds the max chunk_size
    for idx, chunk in enumerate(chunks):
        assert (
            len(chunk.content) <= 150
        ), f"Chunk {idx} exceeded the maximum target size of 150 characters"


def test_metadata_preservation(mock_document: Document) -> None:
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(mock_document)

    # Verify that every chunk carries over the exact parent metadata
    for chunk in chunks:
        assert chunk.metadata.source_file == "test_manual.pdf"
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.total_pages == 5


def test_chunk_ids_are_unique(mock_document: Document) -> None:
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(mock_document)

    # Extract all chunk IDs
    chunk_ids = [chunk.id for chunk in chunks]

    # Verify that all generated IDs are unique
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs were detected!"

    # Verify that the generated ID is a valid, parsing UUID string
    try:
        uuid.UUID(chunks[0].id)
        is_valid_uuid = True
    except ValueError:
        is_valid_uuid = False

    assert is_valid_uuid, f"Chunk ID '{chunks[0].id}' is not a valid UUID string!"
