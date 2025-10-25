from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL

class Embedder:
    """Convert text into dense embeddings using a pre-trained SentenceTransformer model."""

    def __init__(self, device=None):
        """Load the embedding model on the specified device (CPU, CUDA, or MPS)."""
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL} on device {device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    def get_embedding(self, text: str) -> np.ndarray:
        """Return a float32 vector representing the input text embedding."""
        print(f"[Embedder] Embedding text: {text[:60]}...")
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.astype(np.float32)
