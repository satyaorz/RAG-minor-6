import unittest

from hamhrag.agents.validator import AnswerValidator
from hamhrag.models import RetrievedDocument


class AnswerValidatorTest(unittest.TestCase):
    """Validator tests aligned with the simplified _question_alignment design.

    The validator now uses evidence-overlap grounding instead of per-category
    regex checks.  Category-level rejection (wrong entity, wrong water body)
    is delegated to the LLM judge when an LLM client is available.  Without an
    LLM, the heuristic validator only rejects answers that are clearly
    unsupported by the evidence.
    """

    def test_accepts_creation_answer_for_creation_question(self) -> None:
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="Kansas_City_Royals-chunk-3",
                source_type="vector",
                content="The Kansas City Royals were founded as an expansion franchise in 1969.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Kansas_City_Royals-chunk-2",
                source_type="vector",
                content="The Kansas City Royals began play in 1969.",
                score=0.88,
            ),
        ]

        result = validator.validate(
            "The Kansas City Royals were founded in 1969.",
            docs,
            question="When was the baseball team winning the world series in 2015 baseball created?",
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.category_match)

    def test_accepts_answer_grounded_in_requested_body_of_water(self) -> None:
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="Baltic_Sea-chunk-1",
                source_type="vector",
                content=(
                    "The Baltic Sea is bordered by Russian shore areas including "
                    "the Saint Petersburg area and the Kaliningrad Oblast."
                ),
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Baltic_Sea-chunk-2",
                source_type="vector",
                content="The Saint Petersburg area is a Russian shore area of the Baltic Sea.",
                score=0.9,
            ),
        ]

        result = validator.validate(
            "Saint Petersburg borders the Baltic Sea.",
            docs,
            question="Which major Russian city borders Baltic Sea?",
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.category_match)

    def test_rejects_answer_with_no_evidence_overlap(self) -> None:
        """Answer that has zero token overlap with evidence should fail."""
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="d1",
                source_type="vector",
                content="The quick brown fox jumps over the lazy dog.",
                score=0.9,
            ),
        ]

        result = validator.validate(
            "Quantum entanglement superconductors nanotechnology.",
            docs,
            question="What is quantum physics?",
        )

        self.assertFalse(result.passed)

    def test_rejects_no_evidence_phrase(self) -> None:
        """Answers indicating insufficient evidence should always fail."""
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(source_id="d1", source_type="vector",
                              content="Some evidence about topic X.", score=0.9),
            RetrievedDocument(source_id="d2", source_type="vector",
                              content="More evidence about topic X.", score=0.85),
        ]

        result = validator.validate(
            "The evidence does not mention the answer.",
            docs,
            question="What is the answer?",
        )

        self.assertFalse(result.passed)

    def test_accepts_well_grounded_factual_answer(self) -> None:
        """A short factual answer fully grounded in evidence should pass."""
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="d1",
                source_type="vector",
                content="Theodore Roosevelt Sr. was born on September 22, 1831 in New York.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="d2",
                source_type="vector",
                content="Martha Bulloch married Theodore Roosevelt Sr.",
                score=0.8,
            ),
        ]

        result = validator.validate(
            "Theodore Roosevelt Sources: 1, 2",
            docs,
            question="Who is Martha Bulloch Roosevelt's husband?",
        )

        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
