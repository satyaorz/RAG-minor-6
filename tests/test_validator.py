import unittest

from treeqa.agents.validator import AnswerValidator
from treeqa.models import RetrievedDocument


class AnswerValidatorTest(unittest.TestCase):
    def test_rejects_supported_answer_that_misses_creation_category(self) -> None:
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="Kansas_City_Royals-chunk-3",
                source_type="vector",
                content=(
                    "The Kansas City Royals won the 2015 World Series. "
                    "The Kansas City Royals were founded as an expansion franchise in 1969."
                ),
                score=0.95,
            ),
            RetrievedDocument(
                source_id="2015_World_Series-chunk-4",
                source_type="vector",
                content="The Kansas City Royals won the 2015 World Series.",
                score=0.9,
            ),
        ]

        result = validator.validate(
            "The 2015 World Series was won by the Kansas City Royals.",
            docs,
            question="When was the baseball team winning the world series in 2015 baseball created?",
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.category_match)

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

    def test_rejects_answer_grounded_in_wrong_body_of_water(self) -> None:
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="Kuybyshev_Reservoir-chunk-4",
                source_type="vector",
                content="Kazan is a major Russian city adjacent to the Kuybyshev Reservoir.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Kuybyshev_Reservoir-chunk-5",
                source_type="vector",
                content="Other cities adjacent to the Kuybyshev Reservoir include Ulyanovsk.",
                score=0.9,
            ),
        ]

        result = validator.validate(
            "Kazan borders the Kuybyshev Reservoir.",
            docs,
            question="Which major Russian city borders Baltic Sea?",
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.category_match)

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

    def test_rejects_body_of_water_answer_without_water_body(self) -> None:
        validator = AnswerValidator()
        docs = [
            RetrievedDocument(
                source_id="Estonia-chunk-3",
                source_type="vector",
                content="A legend says Tharapita flew to Oesel, Saaremaa from Virumaa.",
                score=0.95,
            ),
            RetrievedDocument(
                source_id="Estonia-chunk-4",
                source_type="vector",
                content="The story has been associated with Kaali crater in Saaremaa.",
                score=0.9,
            ),
        ]

        result = validator.validate(
            "A legend says Tharapita flew to Saaremaa from Virumaa.",
            docs,
            question="Which body of water is Saaremaa located in?",
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.category_match)


if __name__ == "__main__":
    unittest.main()
