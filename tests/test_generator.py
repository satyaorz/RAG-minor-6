import unittest

from treeqa.agents.generator import AnswerGenerator
from treeqa.models import QueryNode, RetrievedDocument


class AnswerGeneratorTest(unittest.TestCase):
    def test_generate_for_node_fallback_adds_sources(self) -> None:
        generator = AnswerGenerator()
        documents = [
            RetrievedDocument(
                source_id="doc-1",
                source_type="vector",
                content="TreeQA validates sub-answers against retrieved evidence.",
                score=0.9,
            )
        ]

        answer = generator.generate_for_node("How does TreeQA validate?", documents)

        self.assertIn("Sources:", answer)
        self.assertIn("vector:doc-1", answer)

    def test_generate_final_fallback_dedupes_and_adds_sources(self) -> None:
        generator = AnswerGenerator()
        nodes = [
            QueryNode(
                node_id="node-1",
                question="Q1",
                answer="Hybrid retrieval combines vector and graph evidence.",
                documents=[
                    RetrievedDocument(
                        source_id="fact-1",
                        source_type="graph",
                        content="Hybrid retrieval combines vector and graph evidence.",
                        score=1.0,
                    )
                ],
            ),
            QueryNode(
                node_id="node-2",
                question="Q2",
                answer="Hybrid retrieval combines vector and graph evidence.",
                documents=[
                    RetrievedDocument(
                        source_id="fact-1",
                        source_type="graph",
                        content="Hybrid retrieval combines vector and graph evidence.",
                        score=1.0,
                    )
                ],
            ),
        ]

        answer = generator.generate_final("What supports TreeQA?", nodes)

        self.assertEqual(answer.count("Hybrid retrieval combines vector and graph evidence."), 1)
        self.assertIn("Sources:", answer)
        self.assertIn("graph:fact-1", answer)

    def test_clean_text_fixes_spacing(self) -> None:
        generator = AnswerGenerator()

        cleaned = generator._clean_text("TreeQA reduces hallucinations . Sources : graph:fact-1")

        self.assertEqual(cleaned, "TreeQA reduces hallucinations. Sources: graph:fact-1")

    def test_strip_sources_removes_trailing_source_block(self) -> None:
        generator = AnswerGenerator()

        stripped = generator._strip_sources(
            "TreeQA uses hybrid retrieval. Sources: vector:doc-1, graph:fact-2"
        )

        self.assertEqual(stripped, "TreeQA uses hybrid retrieval.")


if __name__ == "__main__":
    unittest.main()
