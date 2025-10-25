import numpy as np
import faiss
from src.embeddings import Embedder
from src.config import TOP_K

class Retriever:
    """FAISS retriever for PDF documents."""

    def __init__(self, documents: list, device=None):
        print("[Retriever] Initializing retriever...")
        self.top_k = TOP_K
        self.embedder = Embedder(device=device)
        self.documents = documents
        if not documents:
            raise ValueError("No documents provided for Retriever!")
        self.doc_embeddings = np.array([self.embedder.get_embedding(d) for d in documents], dtype=np.float32)
        self.index = self._build_index(self.doc_embeddings)

    def _build_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        print(f"[Retriever] Building FAISS index with dimension {dim}")
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query: str) -> list:
        print(f"[Retriever] Retrieving top-{self.top_k} documents for query: {query}")
        query_vec = self.embedder.get_embedding(query).astype(np.float32)
        distances, indices = self.index.search(np.array([query_vec]), self.top_k)
        print(f"[Retriever] Distances: {distances[0]}")
        return [self.documents[i] for i in indices[0]]
