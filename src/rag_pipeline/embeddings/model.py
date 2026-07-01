import asyncio
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
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension() or 384
        print(f"[EMBEDDING] Model loaded. Dimensions: {self.dimension}")

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Private synchronous helper that performs the actual model encode.
        This runs the deep-learning matrix multiplications.
        """
        # force return_tensors/numpy to list conversion
        embeddings = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        # Convert numpy array to standard Python list of floats
        return cast(List[List[float]], embeddings.tolist())

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously encodes a list of texts into vector embeddings.
        Offloads the heavy mathematical matrix calculations to a background thread
        to prevent blocking the main Event Loop.
        """
        if not texts:
            return []

        # Use asyncio.to_thread to run self._encode_sync in a background thread
        return await asyncio.to_thread(self._encode_sync, texts)
