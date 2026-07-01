import os
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_pipeline.ingestion.chunker import Chunk


class QdrantStore:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Initializes the asynchronous Qdrant Client.
        Dynamically reads host and port from environment variables to support Docker.
        """
        # 1. Read host and port from environment variables if not passed explicitly
        qdrant_host = host or os.getenv("QDRANT_HOST", "localhost")

        # Read the environment port as a string (completely separate from 'port' parameter)
        env_port = os.getenv("QDRANT_PORT", "6333")

        # If an explicit port was passed, use its string representation; otherwise use env_port
        port_str = str(port) if port is not None else env_port

        try:
            qdrant_port = int(port_str)
        except ValueError:
            qdrant_port = 6333

        self.client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)

    async def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Asynchronously creates a new collection in Qdrant if it does not already exist.
        """
        # 1. Check if the collection already exists to avoid throwing errors
        collections_response = await self.client.get_collections()
        existing_names = [col.name for col in collections_response.collections]

        if collection_name in existing_names:
            print(
                f"[QDRANT] Collection '{collection_name}' already exists. Skipping creation."
            )
            return

        print(
            f"[QDRANT] Creating collection '{collection_name}' with dimension {vector_size}..."
        )

        # 2. Create the collection using Cosine distance as the semantic similarity metric
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"[QDRANT] Collection '{collection_name}' created successfully.")

    async def upsert_chunks(
        self, collection_name: str, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        """
        Converts the Pydantic Chunk objects and calculated vectors into Qdrant Points
        and uploads (upserts) them to the database.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of calculated embeddings!"
            )

        points: List[PointStruct] = []

        for idx, chunk in enumerate(chunks):
            # Extract the raw vector coordinate list
            vector = embeddings[idx]

            # Convert the Pydantic chunk metadata into a standard dictionary payload
            payload: Dict[str, Any] = {
                "text": chunk.content,
                "source_file": chunk.metadata.source_file,
                "page_number": chunk.metadata.page_number,
                "total_pages": chunk.metadata.total_pages,
                "chunk_index": chunk.chunk_index,
            }

            # Create a Qdrant PointStruct (Point Structure)
            point = PointStruct(
                id=chunk.id,  # Uses the unique chunk string ID
                vector=vector,
                payload=payload,
            )
            points.append(point)

        print(f"[QDRANT] Upserting {len(points)} points into '{collection_name}'...")

        # Use self.client.upsert to asynchronously upload the list of points
        await self.client.upsert(collection_name=collection_name, points=points)
        print("[QDRANT] Upsert complete.")
