import os
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        """
        Initializes the SentenceTransformer model with secure HF Token authentication.
        """
        print(f"[EMBEDDING] Loading multilingual model: {model_name}...")
        
        # Fetch HF_TOKEN to authenticate the download and bypass rate limits
        hf_token = os.getenv("HF_TOKEN")
        
        # Pass the token explicitly to the constructor
        self.model = SentenceTransformer(model_name, token=hf_token)
        
        # Correct method name: get_sentence_embedding_dimension()
        self.dimension = self.model.get_embedding_dimension() or 384
        print(f"[EMBEDDING] Model loaded. Dimensions: {self.dimension}")

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Private synchronous helper that performs the actual model encode.
        This runs the deep-learning matrix multiplications.
        """
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        # Convert numpy array to standard Python list of floats
        return embeddings.tolist() # type: ignore

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously encodes a list of texts into vector embeddings.
        Offloads the heavy mathematical matrix calculations to a background thread
        to prevent blocking the main Event Loop.
        """
        import asyncio
        if not texts:
            return []

        # Offload the heavy CPU-bound matrix math to a background thread
        return await asyncio.to_thread(self._encode_sync, texts)