import os
import asyncio
from typing import List, cast

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
