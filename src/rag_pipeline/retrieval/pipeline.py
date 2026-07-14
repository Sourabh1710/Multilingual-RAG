from typing import List, Optional

from rag_pipeline.embeddings.model import EmbeddingModel
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.ingestion.loader import DocumentMetadata
from rag_pipeline.ingestion.preprocess import clean_and_normalize_text
from rag_pipeline.retrieval.store import QdrantStore


class RetrievalPipeline:
    def __init__(self, collection_name: str = "document_chunks", model: Optional[EmbeddingModel] = None) -> None:
        """
        Initializes the retrieval pipeline with the embedding model and database store.
        """
        self.collection_name = collection_name
        self.model = model or EmbeddingModel()
        self.store = QdrantStore()

    async def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        """
        Asynchronously retrieves the top-K most semantically relevant English chunks
        matching a Hindi/Hinglish/English query.
        """
        if not query.strip():
            return []

        # 1. Query Preprocessing: Align bytes using the exact same normalization rules
        normalized_query = clean_and_normalize_text(query)

        # 2. Query Vector Generation: Convert text to coordinates in background thread
        print(f"[RETRIEVAL] Generating query vector for: '{normalized_query}'...")
        query_vectors = await self.model.encode([normalized_query])
        query_vector = query_vectors[0]

        # 3. Perform High-Speed Vector Search in Qdrant
        print(
            f"[RETRIEVAL] Searching Qdrant collection '{self.collection_name}' (top_k={top_k})..."
        )

        # Query Qdrant directly using store's underlying AsyncQdrantClient
        search_results = await self.store.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k
        )

        retrieved_chunks: List[Chunk] = []

        # 4. Map the raw Qdrant Points back into the typed Pydantic Chunk objects
        for point in search_results.points:
            payload = point.payload
            if not payload:
                continue

            # Reconstruct the Pydantic metadata
            metadata = DocumentMetadata(
                source_file=payload.get("source_file", "unknown"),
                page_number=payload.get("page_number", 1),
                total_pages=payload.get("total_pages", 1),
            )

            # Reconstruct the Pydantic Chunk object
            # I cast point.id to string since Qdrant IDs can be returned as UUID strings
            chunk = Chunk(
                id=str(point.id),
                content=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", 0),
                metadata=metadata,
            )

            # Print a clean, helpful log during runtime retrieval
            print(
                f" -> Found match: {metadata.source_file} (Page {metadata.page_number}) | Score: {point.score:.4f}"
            )
            retrieved_chunks.append(chunk)

        print(
            f"[RETRIEVAL] Retrieval complete. Returned {len(retrieved_chunks)} chunks."
        )
        return retrieved_chunks
