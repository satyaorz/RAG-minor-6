"""Tests for the treeqa.eval benchmark runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from treeqa.eval import _exact_match, _token_f1, _load_dataset, run_benchmark
from treeqa.models import PipelineResult, QueryNode


def _make_mock_pipeline(answer: str = "Hybrid retrieval combines vector and graph evidence.") -> MagicMock:
    node = QueryNode(node_id="root", question="q")
    node.answer = answer
    node.status = "verified"
    mock_result = PipelineResult(
        query="q",
        root=node,
        nodes=[node],
        final_answer=answer,
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result
    return mock_pipeline


class EvalMetricsTest(unittest.TestCase):
    def test_exact_match_identical(self) -> None:
        self.assertEqual(_exact_match("HotpotQA", "hotpotqa"), 1)

    def test_exact_match_articles_stripped(self) -> None:
        self.assertEqual(_exact_match("a multi-hop dataset", "multi-hop dataset"), 1)

    def test_exact_match_mismatch(self) -> None:
        self.assertEqual(_exact_match("foo", "bar"), 0)

    def test_token_f1_partial_overlap(self) -> None:
        f1 = _token_f1("multi-hop question answering", "question answering")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_token_f1_perfect(self) -> None:
        self.assertAlmostEqual(_token_f1("hybrid retrieval", "hybrid retrieval"), 1.0)

    def test_token_f1_no_overlap(self) -> None:
        self.assertAlmostEqual(_token_f1("alpha beta", "gamma delta"), 0.0)

    def test_load_dataset_filters_bad_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            lines = [
                '{"question": "What is TreeQA?", "answer": "A RAG system."}',
                '{"question": "No answer field."}',
                '{"answer": "No question field."}',
                "",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")
            records = _load_dataset(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["question"], "What is TreeQA?")


class EvalRunnerTest(unittest.TestCase):
    def test_run_benchmark_writes_results_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "bench.jsonl"
            dataset.write_text(
                '{"question": "What is hybrid retrieval?", "answer": "Combines vector and graph evidence."}\n',
                encoding="utf-8",
            )
            mock_pipeline = _make_mock_pipeline("Combines vector and graph evidence.")
            with patch("treeqa.eval.TreeQAPipeline", return_value=mock_pipeline):
                summary = run_benchmark(dataset, output_dir=tmp_path, limit=1)

            self.assertIn("n", summary)
            self.assertEqual(summary["n"], 1)
            self.assertIn("exact_match", summary)
            self.assertIn("f1", summary)
            results_path = Path(summary["results_path"])
            self.assertTrue(results_path.exists())
            rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertIn("predicted", rows[0])
            self.assertIn("em", rows[0])
            self.assertIn("f1", rows[0])

    def test_run_benchmark_handles_pipeline_error(self) -> None:
        """Pipeline errors should be captured as predicted text, not crash the runner."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "bench.jsonl"
            dataset.write_text(
                '{"question": "Error question?", "answer": "gold"}\n',
                encoding="utf-8",
            )
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("LLM call failed")
            with patch("treeqa.eval.TreeQAPipeline", return_value=mock_pipeline):
                summary = run_benchmark(dataset, output_dir=tmp_path, limit=1)
            self.assertEqual(summary["n"], 1)
            results_path = Path(summary["results_path"])
            rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
            self.assertIn("ERROR", rows[0]["predicted"])


if __name__ == "__main__":
    unittest.main()
