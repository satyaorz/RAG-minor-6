from __future__ import annotations

from treeqa.backends.graph import GraphBackend, MemoryGraphBackend
from treeqa.backends.vector import MemoryVectorBackend, VectorBackend
from treeqa.models import RetrievedDocument


class HybridRetriever:
    """Combines vector and graph retrieval through swappable backends."""

    def __init__(
        self,
        vector_backend: VectorBackend | None = None,
        graph_backend: GraphBackend | None = None,
        top_k: int = 3,
    ) -> None:
        self.vector_backend = vector_backend or MemoryVectorBackend()
        self.graph_backend = graph_backend or MemoryGraphBackend()
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedDocument]:
        documents = self.vector_backend.search(question, self.top_k)
        documents.extend(self.graph_backend.search(question, self.top_k))
        return sorted(documents, key=lambda document: document.score, reverse=True)
