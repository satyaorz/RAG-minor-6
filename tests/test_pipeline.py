"""Offline pipeline tests — no live LLM or file-system dependencies."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from treeqa.agents.decomposer import QueryDecomposer
from treeqa.config import TreeQASettings
from treeqa.models import PipelineResult, QueryNode, RetrievedDocument, ValidationResult
from treeqa.pipeline import TreeQAPipeline


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


def _offline_settings() -> TreeQASettings:
    """Settings that do not trigger any network or file-system access."""
    return TreeQASettings(
        llm_provider="stub",
        vector_provider="memory",
        graph_provider="memory",
    )


def _make_pipeline(**overrides) -> TreeQAPipeline:
    """Build a TreeQAPipeline where every component defaults to a predictable stub."""
    return TreeQAPipeline(
        settings=_offline_settings(),
        decomposer=overrides.get("decomposer", QueryDecomposer(llm_client=None)),
        retriever=overrides.get("retriever", _stub_retriever()),
        validator=overrides.get("validator", _stub_validator()),
        corrector=overrides.get("corrector", _stub_corrector()),
        generator=overrides.get("generator", _stub_generator()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TreeQAPipelineTest(unittest.TestCase):
    def test_pipeline_returns_final_answer_and_tree(self) -> None:
        pipeline = _make_pipeline()

        result = pipeline.run(
            "How does TreeQA use hybrid retrieval and validation for multi-hop QA?"
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


if __name__ == "__main__":
    unittest.main()

