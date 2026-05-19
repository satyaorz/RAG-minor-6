import os
from pathlib import Path
import tempfile
import unittest

from hamhrag.config import HamhRagSettings, load_dotenv


class HamhRagConfigTest(unittest.TestCase):
    def test_load_dotenv_reads_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "HAMHRAG_LLM_PROVIDER=openrouter\n"
                "HAMHRAG_LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free\n",
                encoding="utf-8",
            )

            old_provider = os.environ.pop("HAMHRAG_LLM_PROVIDER", None)
            old_model = os.environ.pop("HAMHRAG_LLM_MODEL", None)
            try:
                load_dotenv(dotenv_path)
                settings = HamhRagSettings.from_env()
                self.assertEqual(settings.llm_provider, "openrouter")
                self.assertEqual(
                    settings.llm_model, "meta-llama/llama-3.3-70b-instruct:free"
                )
            finally:
                os.environ.pop("HAMHRAG_LLM_PROVIDER", None)
                os.environ.pop("HAMHRAG_LLM_MODEL", None)
                if old_provider is not None:
                    os.environ["HAMHRAG_LLM_PROVIDER"] = old_provider
                if old_model is not None:
                    os.environ["HAMHRAG_LLM_MODEL"] = old_model

    def test_load_dotenv_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text("HAMHRAG_LLM_PROVIDER=stub\n", encoding="utf-8")

            old_provider = os.environ.get("HAMHRAG_LLM_PROVIDER")
            try:
                os.environ["HAMHRAG_LLM_PROVIDER"] = "openrouter"
                load_dotenv(dotenv_path)
                settings = HamhRagSettings.from_env()
                self.assertEqual(settings.llm_provider, "openrouter")
            finally:
                if old_provider is None:
                    os.environ.pop("HAMHRAG_LLM_PROVIDER", None)
                else:
                    os.environ["HAMHRAG_LLM_PROVIDER"] = old_provider

    def test_from_env_parses_fallback_models(self) -> None:
        old_fallbacks = os.environ.get("HAMHRAG_LLM_FALLBACK_MODELS")
        try:
            os.environ["HAMHRAG_LLM_FALLBACK_MODELS"] = (
                "arcee-ai/trinity-large-preview:free, openrouter/free"
            )
            settings = HamhRagSettings.from_env()
            self.assertEqual(
                settings.llm_fallback_models,
                ("arcee-ai/trinity-large-preview:free", "openrouter/free"),
            )
        finally:
            if old_fallbacks is None:
                os.environ.pop("HAMHRAG_LLM_FALLBACK_MODELS", None)
            else:
                os.environ["HAMHRAG_LLM_FALLBACK_MODELS"] = old_fallbacks

    def test_resolve_path_uses_project_root_for_relative_paths(self) -> None:
        settings = HamhRagSettings(data_dir="data")

        resolved = settings.resolved_data_dir

        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.name, "data")


if __name__ == "__main__":
    unittest.main()
