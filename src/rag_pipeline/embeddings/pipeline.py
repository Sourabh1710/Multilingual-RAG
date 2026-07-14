from typing import List, Optional

from rag_pipeline.embeddings.model import EmbeddingModel
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.retrieval.store import QdrantStore


class EmbeddingPipeline:
    def __init__(
        self, collection_name: str = "document_chunks", batch_size: int = 16, model: Optional[EmbeddingModel] = None
    ) -> None:
        """
        Initializes the embedding pipeline with the model, Qdrant client, and batch limits.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size

        # Instantiate the modular database store and embedding model wrappers
        self.store = QdrantStore()
        self.model = model or EmbeddingModel()

    async def initialize_database(self) -> None:
        """
        Ensures the target collection is created in Qdrant before it attempt to write to it.
        """
        await self.store.create_collection(
            collection_name=self.collection_name, vector_size=self.model.dimension
        )

    async def embed_and_store(self, chunks: List[Chunk]) -> None:
        """
        Slices chunks into batches, generates embeddings asynchronously,
        and upserts them into Qdrant.
        """
        if not chunks:
            print("[PIPELINE] No chunks provided for embedding.")
            return

        # Ensure collection exists before doing anything
        await self.initialize_database()

        print(f"[PIPELINE] Starting embedding pipeline for {len(chunks)} chunks...")

        all_embeddings: List[List[float]] = []

        # 1. Slice chunks into batches of size self.batch_size
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch]

            print(
                f"[PIPELINE] Embedding batch {i // self.batch_size + 1} ({len(batch_texts)} chunks)..."
            )

            # Generate vectors asynchronously (runs in a background thread)
            batch_embeddings = await self.model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # 2. Upload all chunks and their newly calculated vectors to Qdrant
        print(f"[PIPELINE] Uploading all {len(chunks)} embedded chunks to Qdrant...")
        await self.store.upsert_chunks(
            collection_name=self.collection_name,
            chunks=chunks,
            embeddings=all_embeddings,
        )
        print("[PIPELINE] Pipeline finished. All chunks indexed successfully.")
