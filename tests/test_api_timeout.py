import unittest

from treeqa.api.app import _pipeline_budget, _resolve_timeout_seconds


class ApiTimeoutHelpersTest(unittest.TestCase):
    def test_resolve_timeout_uses_default_when_missing(self) -> None:
        timeout = _resolve_timeout_seconds(None)
        self.assertGreater(timeout, 0.0)

    def test_resolve_timeout_clamps_low_and_high(self) -> None:
        self.assertEqual(_resolve_timeout_seconds(1), 15.0)
        self.assertEqual(_resolve_timeout_seconds(9999), 900.0)

    def test_pipeline_budget_reserves_overhead(self) -> None:
        self.assertEqual(_pipeline_budget(300.0), 298.0)
        self.assertEqual(_pipeline_budget(15.0), 13.0)
        self.assertEqual(_pipeline_budget(1.0), 5.0)


if __name__ == "__main__":
    unittest.main()
