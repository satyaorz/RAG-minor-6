import unittest

from hamhrag.agents.generator import AnswerGenerator
from hamhrag.models import QueryNode, RetrievedDocument, ValidationResult


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.response

    def generate_json(self, system_prompt: str, user_prompt: str):
        return {}


class ExplodingLLM:
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("Final synthesis should not call the LLM for direct bridge answers.")

    def generate_json(self, system_prompt: str, user_prompt: str):
        raise AssertionError("Final synthesis should not call the LLM for direct bridge answers.")


class AnswerGeneratorTest(unittest.TestCase):
    def test_generate_for_node_fallback_adds_sources(self) -> None:
        generator = AnswerGenerator()
        documents = [
            RetrievedDocument(
                source_id="doc-1",
                source_type="vector",
                content="HamhRag validates sub-answers against retrieved evidence.",
                score=0.9,
            )
        ]

        answer = generator.generate_for_node("How does HamhRag validate?", documents)

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

        answer = generator.generate_final("What supports HamhRag?", nodes)

        self.assertEqual(answer.count("Hybrid retrieval combines vector and graph evidence."), 1)
        self.assertIn("Sources:", answer)
        self.assertIn("graph:fact-1", answer)

    def test_clean_text_fixes_spacing(self) -> None:
        generator = AnswerGenerator()

        cleaned = generator._clean_text("HamhRag reduces hallucinations . Sources : graph:fact-1")

        self.assertEqual(cleaned, "HamhRag reduces hallucinations. Sources: graph:fact-1")

    def test_strip_sources_removes_trailing_source_block(self) -> None:
        generator = AnswerGenerator()

        stripped = generator._strip_sources(
            "HamhRag uses hybrid retrieval. Sources: vector:doc-1, graph:fact-2"
        )

        self.assertEqual(stripped, "HamhRag uses hybrid retrieval.")

    def test_generate_final_preserves_terminal_creation_answer(self) -> None:
        generator = AnswerGenerator(
            llm_client=ExplodingLLM()
        )
        nodes = [
            QueryNode(
                node_id="node-1",
                question="When was the baseball team established?",
                answer=(
                    "The baseball team was established in 1969, as stated in the "
                    "evidence regarding the Kansas City Royals' founding. "
                    "Sources: vector:Kansas_City_Royals-chunk-3"
                ),
                documents=[
                    RetrievedDocument(
                        source_id="Kansas_City_Royals-chunk-3",
                        source_type="vector",
                        content="The Kansas City Royals were founded as an expansion franchise in 1969.",
                        score=0.95,
                    )
                ],
                validation=ValidationResult(passed=True, confidence=0.95, rationale="Grounded."),
            ),
            QueryNode(
                node_id="node-2",
                question="What team won the 2015 World Series?",
                answer=(
                    "The 2015 World Series was won by the Kansas City Royals. "
                    "Sources: vector:2015_World_Series-chunk-4"
                ),
                documents=[
                    RetrievedDocument(
                        source_id="2015_World_Series-chunk-4",
                        source_type="vector",
                        content="The Kansas City Royals won the 2015 World Series.",
                        score=0.91,
                    )
                ],
                validation=ValidationResult(passed=True, confidence=0.95, rationale="Grounded."),
            ),
        ]

        answer = generator.generate_final(
            "When was the baseball team winning the world series in 2015 baseball created?",
            nodes,
        )

        self.assertIn("1969", answer)
        self.assertIn("Kansas City Royals", answer)
        self.assertNotIn("2015 World Series was won", answer)

    def test_generate_final_returns_terminal_russian_city_without_llm(self) -> None:
        generator = AnswerGenerator(llm_client=ExplodingLLM())
        nodes = [
            QueryNode(
                node_id="node-1",
                question="Which body of water is Saaremaa located in?",
                answer="Saaremaa is located in the Baltic Sea. Sources: vector:Estonia-chunk-13",
                documents=[
                    RetrievedDocument(
                        source_id="Estonia-chunk-13",
                        source_type="vector",
                        content="Saaremaa is an Estonian island in the Baltic Sea.",
                        score=0.95,
                    )
                ],
                validation=ValidationResult(passed=True, confidence=0.95, rationale="Grounded."),
            ),
            QueryNode(
                node_id="node-2",
                question="Which major Russian city borders Baltic Sea?",
                answer="Saint Petersburg borders the Baltic Sea. Sources: vector:Baltic_Sea-chunk-2",
                documents=[
                    RetrievedDocument(
                        source_id="Baltic_Sea-chunk-2",
                        source_type="vector",
                        content="The Baltic Sea has Russian shore areas including the Saint Petersburg area.",
                        score=0.95,
                    )
                ],
                validation=ValidationResult(passed=True, confidence=0.95, rationale="Grounded."),
            ),
        ]

        answer = generator.generate_final(
            "Which major Russian city borders the body of water in which Saaremaa is located?",
            nodes,
        )

        self.assertIn("Saint Petersburg", answer)
        self.assertIn("Baltic Sea", answer)
        self.assertIn("Sources:", answer)

    def test_generate_for_node_extracts_world_series_winner_from_team_page(self) -> None:
        generator = AnswerGenerator()
        documents = [
            RetrievedDocument(
                source_id="2015_World_Series-chunk-7 [2015 World Series]",
                source_type="vector",
                content=(
                    "Louis Cardinals, or San Francisco Giants as the NL champions. "
                    "The 2015 World Series was the championship series of Major League Baseball's season."
                ),
                score=1.0,
            ),
            RetrievedDocument(
                source_id="Kansas_City_Royals-chunk-3 [Kansas City Royals]",
                source_type="vector",
                content=(
                    "The team was founded as an expansion franchise in 1969, and has "
                    "participated in four World Series, winning in 1985 and 2015."
                ),
                score=0.9,
            ),
        ]

        answer = generator.generate_for_node(
            "Which baseball team won the 2015 World Series?",
            documents,
        )

        self.assertIn("Kansas City Royals", answer)
        self.assertIn("2015 World Series", answer)

    def test_generate_for_node_extracts_russian_city_for_body_of_water(self) -> None:
        generator = AnswerGenerator()
        documents = [
            RetrievedDocument(
                source_id="Baltic_Sea-chunk-1 [Baltic Sea]",
                source_type="vector",
                content=(
                    "Since May 2004, the Baltic Sea has been almost entirely surrounded "
                    "by countries of the European Union. The only remaining non-EU shore "
                    "areas are Russian: the Saint Petersburg area and the exclave of the "
                    "Kaliningrad Oblast."
                ),
                score=0.95,
            )
        ]

        answer = generator.generate_for_node(
            "Which major Russian city borders Baltic Sea?",
            documents,
        )

        self.assertIn("Saint Petersburg", answer)
        self.assertIn("Baltic Sea", answer)

    def test_generate_for_node_extracts_body_of_water_location(self) -> None:
        generator = AnswerGenerator()
        documents = [
            RetrievedDocument(
                source_id="Estonia-chunk-13",
                source_type="vector",
                content="Saaremaa is an Estonian island in the Baltic Sea.",
                score=0.9,
            )
        ]

        answer = generator.generate_for_node(
            "In which body of water is Saaremaa located?",
            documents,
        )

        self.assertIn("Saaremaa", answer)
        self.assertIn("Baltic Sea", answer)


if __name__ == "__main__":
    unittest.main()
