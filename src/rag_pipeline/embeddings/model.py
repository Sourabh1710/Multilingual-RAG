import asyncio
import os
from typing import List, cast

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """
        Initializes the synchronous SentenceTransformer model.
        """
        print(f"[EMBEDDING] Loading multilingual model: {model_name}...")
        hf_token = os.getenv("HF_TOKEN")
        self.model = SentenceTransformer(model_name, token=hf_token)
        self.dimension = self.model.get_embedding_dimension() or 384
        print(f"[EMBEDDING] Model loaded. Dimensions: {self.dimension}")

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Private synchronous helper that performs the actual model encode.
        """
        embeddings = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return cast(List[List[float]], embeddings.tolist())

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously encodes a list of texts into vector embeddings.
        Offloads the heavy computation to a background thread.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)