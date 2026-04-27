from __future__ import annotations

import concurrent.futures

from treeqa.backends.graph import GraphBackend, MemoryGraphBackend
from treeqa.backends.vector import MemoryVectorBackend, VectorBackend
from treeqa.models import RetrievedDocument
from treeqa.retrieval.scoring import rank_documents


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
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def retrieve(self, question: str) -> list[RetrievedDocument]:
        future_vector = self._executor.submit(self.vector_backend.search, question, self.top_k)
        future_graph = self._executor.submit(self.graph_backend.search, question, self.top_k)
        
        documents = future_vector.result()
        documents.extend(future_graph.result())
        
        return rank_documents(question, documents, self.top_k)
