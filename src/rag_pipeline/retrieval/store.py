import os
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_pipeline.ingestion.chunker import Chunk


class QdrantStore:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """
        Initializes the asynchronous Qdrant Client.
        Dynamically reads host and port from environment variables to support Docker.
        """
        # 1. Read host and port from environment variables if not passed explicitly
        qdrant_host = host or os.getenv("QDRANT_HOST", "localhost")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")

        # If the host is a full URL (cloud endpoint), use the 'url' parameter
        if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
            self.client = AsyncQdrantClient(url=qdrant_host, api_key=qdrant_api_key, timeout=15)
        else:
            # Otherwise, use standard host and port parameters for local connections
            port_str = str(port) if port is not None else os.getenv("QDRANT_PORT", "6333")
            try:
                qdrant_port = int(port_str)
            except ValueError:
                qdrant_port = 6333
            
            self.client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key, timeout=15)

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
    
    async def clear_collection(self, collection_name: str) -> None:
        """
        Asynchronously deletes a collection if it exists, clearing all old points.
        """
        collections_response = await self.client.get_collections()
        existing_names = [col.name for col in collections_response.collections]

        if collection_name in existing_names:
            print(f"[QDRANT] Clearing old collection '{collection_name}' to prevent data pollution...")
            await self.client.delete_collection(collection_name)
            print(f"[QDRANT] Old collection '{collection_name}' cleared successfully.")
