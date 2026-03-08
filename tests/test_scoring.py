import unittest

from treeqa.models import RetrievedDocument
from treeqa.retrieval.scoring import normalize_text, rank_documents, select_relevant_snippet


class RetrievalScoringTest(unittest.TestCase):
    def test_normalize_text_removes_markdown_heading(self) -> None:
        normalized = normalize_text("# TreeQA Overview\n\nUses hybrid retrieval.")

        self.assertEqual(normalized, "TreeQA Overview Uses hybrid retrieval.")

    def test_select_relevant_snippet_prefers_matching_sentences(self) -> None:
        content = (
            "TreeQA is a system for multi-hop QA. "
            "Hybrid retrieval combines vector evidence and graph support. "
            "The UI shows a logic tree."
        )

        snippet = select_relevant_snippet(content, "How does hybrid retrieval work?")

        self.assertIn("Hybrid retrieval combines vector evidence and graph support.", snippet)

    def test_rank_documents_dedupes_same_content(self) -> None:
        documents = [
            RetrievedDocument(
                source_id="doc-1",
                source_type="vector",
                content="Hybrid retrieval combines vector evidence and graph support.",
                score=0.9,
            ),
            RetrievedDocument(
                source_id="fact-1",
                source_type="graph",
                content="Hybrid retrieval combines vector evidence and graph support.",
                score=0.8,
            ),
        ]

        ranked = rank_documents("How does hybrid retrieval work?", documents, 3)

        self.assertEqual(len(ranked), 1)


if __name__ == "__main__":
    unittest.main()
