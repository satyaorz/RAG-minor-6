import unittest

from treeqa.retrieval import HybridRetriever


class HybridRetrieverTest(unittest.TestCase):
    def test_retriever_returns_ranked_documents(self) -> None:
        retriever = HybridRetriever()

        documents = retriever.retrieve("How does hybrid retrieval work in TreeQA?")

        self.assertTrue(documents)
        self.assertEqual(
            documents,
            sorted(documents, key=lambda document: document.score, reverse=True),
        )

    def test_retriever_uses_supported_source_types(self) -> None:
        retriever = HybridRetriever()

        documents = retriever.retrieve("Tell me about TreeQA logic-tree reasoning and Neo4j")

        source_types = {document.source_type for document in documents}
        self.assertTrue(source_types <= {"graph", "vector"})
        self.assertTrue(source_types)


if __name__ == "__main__":
    unittest.main()
