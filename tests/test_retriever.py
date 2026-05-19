import unittest

from hamhrag.models import RetrievedDocument
from hamhrag.retrieval import HybridRetriever


class HybridRetrieverTest(unittest.TestCase):
    def test_retriever_returns_ranked_documents(self) -> None:
        retriever = HybridRetriever()

        documents = retriever.retrieve("How does hybrid retrieval work in HamhRag?")

        self.assertTrue(documents)
        self.assertEqual(
            documents,
            sorted(documents, key=lambda document: document.score, reverse=True),
        )

    def test_retriever_uses_supported_source_types(self) -> None:
        retriever = HybridRetriever()

        documents = retriever.retrieve("Tell me about HamhRag logic-tree reasoning and Neo4j")

        source_types = {document.source_type for document in documents}
        self.assertTrue(source_types <= {"graph", "vector"})
        self.assertTrue(source_types)

    def test_retriever_returns_trace_metadata(self) -> None:
        retriever = HybridRetriever()

        documents, trace = retriever.retrieve_with_trace(
            "Explain how hybrid retrieval combines vector search and graph lookups in HamhRag."
        )

        self.assertTrue(documents)
        self.assertIn(trace.route, {"vector_only", "vector_first", "hybrid_parallel", "graph_first"})
        self.assertGreaterEqual(trace.total_latency_ms, 0.0)
        self.assertTrue(trace.backends_used)

    def test_router_prefers_parallel_for_comparison_queries(self) -> None:
        retriever = HybridRetriever()

        _documents, trace = retriever.retrieve_with_trace(
            "Compare the difference between vector retrieval and graph retrieval in HamhRag."
        )

        self.assertEqual(trace.route, "hybrid_parallel")

    def test_retriever_prunes_off_topic_graph_docs_for_entity_questions(self) -> None:
        class _VectorBackend:
            def search(self, question: str, limit: int) -> list[RetrievedDocument]:
                return [
                    RetrievedDocument(
                        source_id="The_Star_of_Santa_Clara-chunk-1",
                        source_type="vector",
                        content="The Star of Santa Clara is a West German film directed by Werner Jacobs.",
                        score=0.9,
                    ),
                    RetrievedDocument(
                        source_id="Werner_Jacobs-chunk-1",
                        source_type="vector",
                        content="Werner Jacobs was a German film director.",
                        score=0.85,
                    ),
                ][:limit]

        class _GraphBackend:
            def search(self, question: str, limit: int) -> list[RetrievedDocument]:
                return [
                    RetrievedDocument(
                        source_id="fact-tree-2",
                        source_type="graph",
                        content="Hybrid retrieval combines vector evidence with graph-backed support.",
                        score=0.95,
                    )
                ][:limit]

        retriever = HybridRetriever(
            vector_backend=_VectorBackend(),
            graph_backend=_GraphBackend(),
            top_k=3,
        )

        docs, _trace = retriever.retrieve_with_trace(
            "What country is the director of The Star Of Santa Clara from?"
        )

        self.assertTrue(any(doc.source_type == "vector" for doc in docs))
        self.assertFalse(any(doc.source_id == "fact-tree-2" for doc in docs))


if __name__ == "__main__":
    unittest.main()
