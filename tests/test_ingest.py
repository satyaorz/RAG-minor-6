import tempfile
from pathlib import Path
import unittest

from treeqa.backends.graph import LocalGraphBackend
from treeqa.backends.vector import LocalVectorBackend
from treeqa.config import TreeQASettings
from treeqa.ingest import build_local_indices


class TreeQAIngestTest(unittest.TestCase):
    def test_ingest_builds_local_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "documents").mkdir(parents=True)
            (data_dir / "graph").mkdir(parents=True)
            (data_dir / "documents" / "overview.md").write_text(
                "TreeQA uses logic tree reasoning and hybrid retrieval.",
                encoding="utf-8",
            )
            (data_dir / "graph" / "facts.jsonl").write_text(
                '{"source_id":"fact-1","content":"TreeQA validates intermediate steps."}\n',
                encoding="utf-8",
            )

            report = build_local_indices(TreeQASettings(data_dir=str(data_dir)))

            self.assertGreater(report.vector_chunks, 0)
            self.assertGreater(report.graph_facts, 0)
            self.assertTrue(Path(report.vector_index_path).exists())
            self.assertTrue(Path(report.graph_index_path).exists())

    def test_local_backends_read_ingested_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "documents").mkdir(parents=True)
            (data_dir / "graph").mkdir(parents=True)
            (data_dir / "documents" / "overview.md").write_text(
                "TreeQA uses logic tree reasoning and hybrid retrieval.",
                encoding="utf-8",
            )
            (data_dir / "graph" / "facts.jsonl").write_text(
                '{"source_id":"fact-1","content":"TreeQA validates intermediate steps."}\n',
                encoding="utf-8",
            )

            report = build_local_indices(TreeQASettings(data_dir=str(data_dir)))
            vector_backend = LocalVectorBackend(report.vector_index_path)
            graph_backend = LocalGraphBackend(report.graph_index_path)

            self.assertTrue(vector_backend.search("logic tree retrieval", 3))
            self.assertTrue(graph_backend.search("validates intermediate steps", 3))
