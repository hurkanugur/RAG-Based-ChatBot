from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL
import torch

class Embedder:
    """Local embeddings using sentence-transformers."""

    def __init__(self, device=None):
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL} on device {device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    def get_embedding(self, text: str) -> np.ndarray:
        print(f"[Embedder] Embedding text: {text[:60]}...")
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.astype(np.float32)
