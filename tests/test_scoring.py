import unittest

from hamhrag.models import RetrievedDocument
from hamhrag.retrieval.scoring import normalize_text, rank_documents, select_relevant_snippet


class RetrievalScoringTest(unittest.TestCase):
    def test_normalize_text_removes_markdown_heading(self) -> None:
        normalized = normalize_text("# HamhRag Overview\n\nUses hybrid retrieval.")

        self.assertEqual(normalized, "HamhRag Overview Uses hybrid retrieval.")

    def test_select_relevant_snippet_prefers_matching_sentences(self) -> None:
        content = (
            "HamhRag is a system for multi-hop QA. "
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

    def test_rank_documents_boosts_named_entity_match(self) -> None:
        documents = [
            RetrievedDocument(
                source_id="Banderas_River-chunk-1",
                source_type="vector",
                content="Banderas River is located in a department with municipal water supply.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Estonia-chunk-13",
                source_type="vector",
                content="Saaremaa is an Estonian island in the Baltic Sea.",
                score=0.4,
            ),
        ]

        ranked = rank_documents("In which body of water is Saaremaa located?", documents, 2)

        self.assertEqual(ranked[0].source_id, "Estonia-chunk-13")

    def test_rank_documents_boosts_requested_answer_type(self) -> None:
        documents = [
            RetrievedDocument(
                source_id="Estonia-chunk-3",
                source_type="vector",
                content="A legend says Tharapita flew to Oesel, Saaremaa from Virumaa.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Estonia-chunk-12",
                source_type="vector",
                content="Saaremaa is an Estonian island in the Baltic Sea.",
                score=0.4,
            ),
        ]

        ranked = rank_documents("Which body of water is Saaremaa located in?", documents, 2)

        self.assertEqual(ranked[0].source_id, "Estonia-chunk-12")


if __name__ == "__main__":
    unittest.main()
