import unittest

from treeqa.config import TreeQASettings
from treeqa.diagnostics import _check_llm, run_diagnostics


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.response

    def generate_json(self, system_prompt: str, user_prompt: str) -> object:
        return {}


class TreeQADiagnosticsTest(unittest.TestCase):
    def test_diagnostics_succeeds_for_memory_backends(self) -> None:
        settings = TreeQASettings(
            llm_provider="stub",
            vector_provider="memory",
            graph_provider="memory",
        )

        report = run_diagnostics(settings=settings, live_llm_probe=False)

        self.assertTrue(report.ok)
        self.assertEqual(len(report.checks), 3)
        self.assertTrue(all(check.ok for check in report.checks))

    def test_diagnostics_fails_for_invalid_qdrant_config(self) -> None:
        settings = TreeQASettings(
            llm_provider="stub",
            vector_provider="qdrant",
            graph_provider="memory",
        )

        report = run_diagnostics(settings=settings, live_llm_probe=False)

        vector_checks = [check for check in report.checks if check.name == "vector"]
        self.assertEqual(len(vector_checks), 1)
        self.assertFalse(vector_checks[0].ok)

    def test_diagnostics_fails_for_missing_local_index(self) -> None:
        settings = TreeQASettings(
            llm_provider="stub",
            vector_provider="local",
            graph_provider="memory",
            data_dir="missing-data-dir",
        )

        report = run_diagnostics(settings=settings, live_llm_probe=False)

        vector_checks = [check for check in report.checks if check.name == "vector"]
        self.assertEqual(len(vector_checks), 1)
        self.assertFalse(vector_checks[0].ok)

    def test_diagnostics_fails_for_empty_local_indices(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir(parents=True)
            (index_dir / "vector_index.jsonl").write_text("", encoding="utf-8")
            (index_dir / "graph_facts.jsonl").write_text("", encoding="utf-8")

            settings = TreeQASettings(
                llm_provider="stub",
                vector_provider="local",
                graph_provider="local",
                data_dir=temp_dir,
            )

            report = run_diagnostics(settings=settings, live_llm_probe=False)

            self.assertFalse(report.ok)
            self.assertEqual(
                {check.name for check in report.checks if not check.ok},
                {"vector", "graph"},
            )

    def test_llm_live_probe_accepts_non_literal_nonempty_response(self) -> None:
        settings = TreeQASettings(
            llm_provider="openrouter",
            llm_model="arcee-ai/trinity-large-preview:free",
            openrouter_api_key="test-key",
        )

        result = _check_llm_with_fake_client(settings, "Test")

        self.assertTrue(result.ok)
        self.assertIn("reached the model successfully", result.detail)

    def test_llm_live_probe_fails_on_empty_response(self) -> None:
        settings = TreeQASettings(
            llm_provider="openrouter",
            llm_model="arcee-ai/trinity-large-preview:free",
            openrouter_api_key="test-key",
        )

        result = _check_llm_with_fake_client(settings, "")

        self.assertFalse(result.ok)
        self.assertIn("empty response", result.detail)


def _check_llm_with_fake_client(settings: TreeQASettings, response: str):
    from unittest.mock import patch

    fake_client = FakeLLMClient(response=response)
    with patch("treeqa.diagnostics.build_llm_client", return_value=fake_client):
        return _check_llm(settings, live_probe=True)


if __name__ == "__main__":
    unittest.main()
