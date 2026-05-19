"""Offline pipeline tests — no live LLM or file-system dependencies."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from hamhrag.agents.decomposer import QueryDecomposer
from hamhrag.config import HamhRagSettings
from hamhrag.models import PipelineResult, QueryNode, RetrievedDocument, ValidationResult
from hamhrag.pipeline import HamhRagPipeline
from hamhrag.retrieval.hybrid import RetrievalTrace


# ---------------------------------------------------------------------------
# Reusable stubs
# ---------------------------------------------------------------------------

def _stub_two_docs() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(source_id="doc-1", source_type="vector", content="HotpotQA is a multi-hop QA benchmark.", score=0.8),
        RetrievedDocument(source_id="doc-2", source_type="graph", content="LangGraph orchestrates multi-step agent workflows.", score=0.7),
    ]


def _stub_retriever() -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = _stub_two_docs()
    return mock


def _stub_validator(passed: bool = True, confidence: float = 0.85) -> MagicMock:
    mock = MagicMock()
    mock.validate.return_value = ValidationResult(
        passed=passed, confidence=confidence, rationale="Grounded in retrieved evidence."
    )
    return mock


def _stub_generator(answer: str = "Evidence supports this answer.") -> MagicMock:
    mock = MagicMock()
    mock.generate_for_node.return_value = answer
    mock.generate_final.return_value = answer
    return mock


def _stub_corrector() -> MagicMock:
    mock = MagicMock()
    mock.refine.side_effect = lambda question, attempt: f"Refined attempt {attempt}: {question}"
    return mock


def _stub_restructurer(new_nodes: list[QueryNode] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.restructure.return_value = new_nodes
    return mock


def _offline_settings() -> HamhRagSettings:
    """Settings that do not trigger any network or file-system access."""
    return HamhRagSettings(
        llm_provider="stub",
        vector_provider="memory",
        graph_provider="memory",
        max_restructures=1,
    )


def _make_pipeline(**overrides) -> HamhRagPipeline:
    """Build a HamhRagPipeline where every component defaults to a predictable stub."""
    return HamhRagPipeline(
        settings=_offline_settings(),
        decomposer=overrides.get("decomposer", QueryDecomposer(llm_client=None)),
        retriever=overrides.get("retriever", _stub_retriever()),
        validator=overrides.get("validator", _stub_validator()),
        corrector=overrides.get("corrector", _stub_corrector()),
        generator=overrides.get("generator", _stub_generator()),
        restructurer=overrides.get("restructurer", _stub_restructurer()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class HamhRagPipelineTest(unittest.TestCase):
    def test_pipeline_returns_final_answer_and_tree(self) -> None:
        pipeline = _make_pipeline()

        result = pipeline.run(
            "How does HamhRag use hybrid retrieval and validation for multi-hop QA?"
        )

        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.final_answer)
        self.assertEqual(result.root.node_id, "root")
        self.assertIn(result.root.status, {"verified", "needs_review"})
        self.assertTrue(result.nodes)

    def test_pipeline_marks_root_verified_when_validator_passes(self) -> None:
        pipeline = _make_pipeline(validator=_stub_validator(passed=True, confidence=0.9))

        result = pipeline.run("What is HotpotQA?")

        self.assertEqual(result.root.status, "verified")

    def test_pipeline_marks_root_needs_review_when_validator_fails(self) -> None:
        # Validator always fails → max_retries exhausted → needs_review
        pipeline = _make_pipeline(
            validator=_stub_validator(passed=False, confidence=0.1),
        )

        result = pipeline.run("What is HotpotQA?")

        self.assertEqual(result.root.status, "needs_review")
        self.assertGreaterEqual(result.root.attempts, 1)

    def test_pipeline_decomposes_multi_part_query_under_root(self) -> None:
        # Rule-based decomposer splits on " and "
        pipeline = _make_pipeline()

        result = pipeline.run(
            "What is HotpotQA and how does LangGraph support agent workflows?"
        )

        self.assertEqual(result.root.node_id, "root")
        self.assertEqual(len(result.root.children), 2)
        self.assertEqual(len(result.nodes), 2)

    def test_pipeline_uses_root_as_leaf_for_single_part_query(self) -> None:
        pipeline = _make_pipeline()

        result = pipeline.run("What is HotpotQA?")

        self.assertFalse(result.root.children)
        self.assertEqual(result.nodes[0].node_id, "root")

    def test_pipeline_retrieves_documents_for_each_leaf(self) -> None:
        retriever = _stub_retriever()
        pipeline = _make_pipeline(retriever=retriever)

        pipeline.run(
            "What is HotpotQA and how does LangGraph support agent workflows?"
        )

        # Two leaf nodes → retriever called at least twice
        self.assertGreaterEqual(retriever.retrieve.call_count, 2)

    def test_pipeline_records_attempt_count_on_leaf(self) -> None:
        pipeline = _make_pipeline()

        result = pipeline.run("What is HotpotQA?")

        self.assertGreaterEqual(result.root.attempts, 1)

    def test_corrector_called_on_retry(self) -> None:
        corrector = MagicMock()
        corrector.refine.return_value = "Refined: What is HotpotQA?"

        failing_then_passing_validator = MagicMock()
        failing_then_passing_validator.validate.side_effect = [
            ValidationResult(passed=False, confidence=0.1, rationale="Insufficient evidence."),
            ValidationResult(passed=True, confidence=0.9, rationale="Grounded."),
        ]

        pipeline = _make_pipeline(
            validator=failing_then_passing_validator,
            corrector=corrector,
        )
        result = pipeline.run("What is HotpotQA?")

        corrector.refine.assert_called_once()
        self.assertEqual(result.root.status, "verified")
        self.assertEqual(result.root.attempts, 2)

    def test_pipeline_restructures_node_on_failure(self) -> None:
        # Validator always fails
        failing_validator = MagicMock()
        failing_validator.validate.return_value = ValidationResult(
            passed=False, confidence=0.1, rationale="Dead end."
        )

        # Restructurer provides new nodes
        restructured_nodes = [
            QueryNode(node_id="sub-1", question="Splinter A"),
            QueryNode(node_id="sub-2", question="Splinter B"),
        ]
        restructurer = _stub_restructurer(new_nodes=restructured_nodes)

        pipeline = _make_pipeline(
            validator=failing_validator,
            restructurer=restructurer,
        )
        
        # We need to make the validator pass for the new sub-nodes 
        # so the test terminates nicely.
        failing_validator.validate.side_effect = [
            ValidationResult(passed=False, confidence=0.1, rationale="Dead end."), # original leaf fail
            ValidationResult(passed=False, confidence=0.1, rationale="Dead end."), # retry 1 fail
            ValidationResult(passed=False, confidence=0.1, rationale="Dead end."), # retry 2 fail
            ValidationResult(passed=True, confidence=1.0, rationale="Pass A"),      # sub-1 pass
            ValidationResult(passed=True, confidence=1.0, rationale="Pass B"),      # sub-2 pass
        ]

        result = pipeline.run("Complex Question")

        self.assertTrue(result.root.was_restructured)
        self.assertEqual(len(result.root.children), 2)
        restructurer.restructure.assert_called_once()
        self.assertEqual(result.root.status, "verified")

    def test_pipeline_respects_max_retries_override(self) -> None:
        failing_validator = _stub_validator(passed=False, confidence=0.1)
        pipeline = _make_pipeline(validator=failing_validator)

        result = pipeline.run("What is HotpotQA?", max_retries_override=0)

        self.assertEqual(result.root.attempts, 1)

    def test_pipeline_respects_tree_retries_override(self) -> None:
        corrector = MagicMock()
        corrector.refine.return_value = "Refined once"
        pipeline = _make_pipeline(
            validator=_stub_validator(passed=True, confidence=0.95),
            corrector=corrector,
        )

        pipeline.run("What is HotpotQA?", tree_retries_override=0)

        corrector.refine.assert_not_called()

    def test_pipeline_respects_max_restructures_override(self) -> None:
        failing_validator = _stub_validator(passed=False, confidence=0.1)
        restructurer = _stub_restructurer(
            new_nodes=[QueryNode(node_id="sub-1", question="A"), QueryNode(node_id="sub-2", question="B")]
        )
        pipeline = _make_pipeline(
            validator=failing_validator,
            restructurer=restructurer,
        )

        pipeline.run("Complex Question", max_restructures_override=0)

        restructurer.restructure.assert_not_called()

    def test_pipeline_bridges_director_country_entity(self) -> None:
        class _Retriever:
            def __init__(self) -> None:
                self.top_k = 3
                self.vector_backend = self

            def retrieve_with_trace(self, question: str):
                docs = [
                    RetrievedDocument(
                        source_id="The_Star_of_Santa_Clara-chunk-1",
                        source_type="vector",
                        content="The Star of Santa Clara is a 1958 film directed by Werner Jacobs.",
                        score=0.9,
                    )
                ]
                trace = RetrievalTrace(
                    route="graph_first",
                    query_type="entity_relation",
                    reason="test",
                    signals={},
                    vector_limit=3,
                    graph_limit=3,
                    backends_used=["vector"],
                )
                return docs, trace

            def retrieve(self, question: str):
                docs, _trace = self.retrieve_with_trace(question)
                return docs

            def search(self, question: str, limit: int) -> list[RetrievedDocument]:
                return [
                    RetrievedDocument(
                        source_id="Werner_Jacobs-chunk-1",
                        source_type="vector",
                        content="Werner Jacobs was a German film director.",
                        score=0.88,
                    )
                ][:limit]

        def _validate(answer: str, docs: list[RetrievedDocument]) -> ValidationResult:
            has_country_evidence = any(
                "german film director" in d.content.lower()
                for d in docs
            )
            return ValidationResult(
                passed=has_country_evidence,
                confidence=0.9 if has_country_evidence else 0.1,
                rationale="Grounded." if has_country_evidence else "Insufficient evidence.",
            )

        validator = MagicMock()
        validator.validate.side_effect = _validate
        generator = MagicMock()
        generator.generate_for_node.return_value = (
            "Werner Jacobs was a German film director. Sources: vector:Werner_Jacobs-chunk-1"
        )
        generator.generate_final.return_value = (
            "No, one director is American and the other is German. Sources: vector:Werner_Jacobs-chunk-1"
        )

        decomposer = MagicMock()
        decomposer.decompose.return_value = QueryNode(
            node_id="root",
            question="What country is the director of The Star Of Santa Clara from?",
        )
        
        pipeline = _make_pipeline(
            decomposer=decomposer,
            retriever=_Retriever(),
            validator=validator,
            generator=generator,
            corrector=_stub_corrector(),
        )

        result = pipeline.run(
            "What country is the director of The Star Of Santa Clara from?",
            max_retries_override=0,
            max_restructures_override=0,
            tree_retries_override=0,
        )

        self.assertEqual(result.root.status, "verified")
        self.assertIn("vector_entity_bridge", result.root.retrieval_backends)

    def test_pipeline_does_not_leak_sibling_answer_into_independent_query(self) -> None:
        class _Retriever:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def retrieve(self, question: str) -> list[RetrievedDocument]:
                self.queries.append(question)
                return [
                    RetrievedDocument(
                        source_id="doc-1",
                        source_type="vector",
                        content=f"Evidence for: {question}",
                        score=0.9,
                    )
                ]

        decomposer = MagicMock()
        decomposer.decompose.return_value = QueryNode(
            node_id="root",
            question="Combined query",
            children=[
                QueryNode(node_id="node-1", question="Who directed Film A?"),
                QueryNode(node_id="node-2", question="What country is the director of Film B from?"),
            ],
        )

        retriever = _Retriever()
        validator = _stub_validator(passed=True, confidence=0.95)
        generator = MagicMock()
        generator.generate_for_node.side_effect = [
            "Director A. Sources: vector:doc-1",
            "Director B is from Country B. Sources: vector:doc-1",
        ]
        generator.generate_final.return_value = "Final"

        pipeline = _make_pipeline(
            decomposer=decomposer,
            retriever=retriever,
            validator=validator,
            generator=generator,
        )

        pipeline.run("Combined query", tree_retries_override=0, max_retries_override=0)

        self.assertEqual(retriever.queries[0], "Who directed Film A?")
        self.assertEqual(
            retriever.queries[1],
            "What country is the director of Film B from?",
        )

    def test_pipeline_resolves_anaphoric_body_of_water_sequentially(self) -> None:
        class _Retriever:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def retrieve(self, question: str) -> list[RetrievedDocument]:
                self.queries.append(question)
                if "Saaremaa" in question:
                    return [
                        RetrievedDocument(
                            source_id="Estonia-chunk-13",
                            source_type="vector",
                            content="Saaremaa is an Estonian island in the Baltic Sea.",
                            score=0.9,
                        )
                    ]
                return [
                    RetrievedDocument(
                        source_id="Baltic_Sea-chunk-1",
                        source_type="vector",
                        content=(
                            "The Baltic Sea has Russian shore areas including "
                            "the Saint Petersburg area."
                        ),
                        score=0.9,
                    )
                ]

        decomposer = MagicMock()
        decomposer.decompose.return_value = QueryNode(
            node_id="root",
            question="Which major Russian city borders the body of water in which Saaremaa is located?",
            children=[
                QueryNode(node_id="node-1", question="In which body of water is Saaremaa located?"),
                QueryNode(node_id="node-2", question="Which major Russian city borders this body of water?"),
            ],
        )

        retriever = _Retriever()
        validator = _stub_validator(passed=True, confidence=0.95)
        generator = MagicMock()
        generator.generate_for_node.side_effect = [
            "Saaremaa is located in the Baltic Sea. Sources: vector:Estonia-chunk-13",
            "Saint Petersburg borders the Baltic Sea. Sources: vector:Baltic_Sea-chunk-1",
        ]
        generator.generate_final.return_value = (
            "Saint Petersburg borders the Baltic Sea. Sources: vector:Baltic_Sea-chunk-1"
        )

        pipeline = _make_pipeline(
            decomposer=decomposer,
            retriever=retriever,
            validator=validator,
            generator=generator,
        )

        result = pipeline.run(
            "Which major Russian city borders the body of water in which Saaremaa is located?",
            tree_retries_override=0,
            max_retries_override=0,
        )

        self.assertEqual(result.root.status, "verified")
        self.assertEqual(retriever.queries[0], "In which body of water is Saaremaa located?")
        self.assertTrue(retriever.queries[1].startswith("Which major Russian city borders Baltic Sea?"))
        self.assertIn("Saaremaa is located in the Baltic Sea.", retriever.queries[1])

    def test_pipeline_resolves_musique_question(self) -> None:
        # e.g., "What record label does the singer in 4 non blondes sign?"
        decomposer = QueryDecomposer(llm_client=None)
        
        class _Retriever:
            def retrieve(self, question: str) -> list[RetrievedDocument]:
                if "singer in 4 non blondes" in question.lower():
                    return [RetrievedDocument(source_id="4NB", source_type="vector", content="Linda Perry was the singer in 4 Non Blondes.", score=0.9)]
                return [RetrievedDocument(source_id="LP", source_type="vector", content="Linda Perry is signed to Custard Records.", score=0.9)]
        
        validator = _stub_validator(passed=True, confidence=0.9)
        generator = MagicMock()
        generator.generate_for_node.side_effect = [
            "Linda Perry. Sources: vector:4NB",
            "Custard Records. Sources: vector:LP",
        ]
        generator.generate_final.return_value = "Custard Records"
        
        pipeline = _make_pipeline(
            decomposer=decomposer,
            retriever=_Retriever(),
            validator=validator,
            generator=generator,
        )
        
        result = pipeline.run("What record label does the singer in 4 non blondes sign?", tree_retries_override=0)
        self.assertEqual(result.root.status, "verified")
        self.assertEqual(result.final_answer, "Custard Records")


if __name__ == "__main__":
    unittest.main()
