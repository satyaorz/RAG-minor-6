"""Tests for LocalVectorBackend (IDF / hybrid) and FaissVectorBackend.

All tests are offline — no live model is loaded.  A tiny synthetic embedding
index is built in memory and, for the FAISS tests, written to a temp directory.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — build tiny synthetic assets without sentence-transformers
# ---------------------------------------------------------------------------

_DIM = 8  # small dimension for unit tests


def _make_docs(n: int = 5) -> list[dict]:
    """Return n synthetic document dicts with distinct content."""
    topics = [
        "Sruthilayalu is a Telugu film composed by K. V. Mahadevan.",
        "K. V. Mahadevan was born on 17 October 1918 in Chennai.",
        "Hybrid retrieval combines dense vectors and sparse lexical signals.",
        "LangGraph helps orchestrate multi-step agent workflows.",
        "FAISS is an efficient library for similarity search.",
    ]
    return [
        {
            "source_id": f"doc-{i}",
            "title": f"Title {i}",
            "section": "Main",
            "chunk_index": i,
            "ingested_at": "2026-01-01T00:00:00",
            "content": topics[i % len(topics)],
        }
        for i in range(n)
    ]


def _make_embeddings(n: int = 5) -> np.ndarray:
    """Return n distinct L2-normalised float32 vectors of length _DIM."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, _DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _fake_st_model(docs: list[dict], embeddings: np.ndarray):
    """Return a SentenceTransformer mock that returns pre-built embeddings."""
    model = MagicMock()
    # encode(texts) maps each call to rows of `embeddings` in order
    model.encode.side_effect = lambda texts, **kw: embeddings[: len(texts)]
    return model


# ---------------------------------------------------------------------------
# LocalVectorBackend — IDF and hybrid search
# ---------------------------------------------------------------------------

class LocalVectorBackendIDFTest(unittest.TestCase):
    """Tests IDF table construction and lexical search without live model."""

    def _make_backend(self):
        from treeqa.backends.vector import LocalVectorBackend

        docs = _make_docs(5)
        embeddings = _make_embeddings(5)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")
            index_path = f.name

        backend = LocalVectorBackend.__new__(LocalVectorBackend)
        backend.index_path = Path(index_path)
        backend.documents = docs
        backend._idf = backend._build_idf()
        backend._model = _fake_st_model(docs, embeddings)
        backend._doc_embeddings = embeddings
        return backend

    def test_idf_table_built_for_all_documents(self) -> None:
        backend = self._make_backend()
        self.assertGreater(len(backend._idf), 0)

    def test_idf_rare_token_scores_higher_than_common(self) -> None:
        """'sruthilayalu' appears in 1 doc; 'mahadevan' in 2 — IDF(rare) > IDF(common)."""
        backend = self._make_backend()
        idf_rare = backend._idf.get("sruthilayalu", 0.0)
        idf_common = backend._idf.get("mahadevan", 0.0)
        # 'sruthilayalu' is in 1 doc; 'mahadevan' is in 2 docs → IDF(rare) > IDF(common)
        self.assertGreater(idf_rare, idf_common)

    def test_lexical_search_surfaces_rare_entity(self) -> None:
        """Querying a rare proper noun should rank the matching doc at position 0."""
        backend = self._make_backend()
        # doc-0 contains 'sruthilayalu', doc-1 contains 'mahadevan'
        results = backend._lexical_search("Sruthilayalu film", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0].source_id, "doc-0 [Main]")

    def test_lexical_search_empty_query_returns_nothing(self) -> None:
        backend = self._make_backend()
        results = backend._lexical_search("   ", limit=5)
        self.assertEqual(results, [])

    def test_hybrid_search_returns_limit_results(self) -> None:
        backend = self._make_backend()
        results = backend._hybrid_search("film composer birthday", limit=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

    def test_hybrid_search_no_duplicate_source_ids(self) -> None:
        backend = self._make_backend()
        results = backend._hybrid_search("hybrid retrieval langraph faiss", limit=5)
        ids = [r.source_id for r in results]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate source_ids in hybrid results")

    def test_search_dispatches_to_hybrid_when_model_present(self) -> None:
        backend = self._make_backend()
        with patch.object(backend, "_hybrid_search", wraps=backend._hybrid_search) as spy:
            backend.search("test query", limit=3)
        spy.assert_called_once_with("test query", 3)


# ---------------------------------------------------------------------------
# FaissVectorBackend
# ---------------------------------------------------------------------------

class FaissVectorBackendTest(unittest.TestCase):
    """Tests FaissVectorBackend with a tiny synthetic FAISS IndexFlatIP.

    Backends are built via __new__ + manual attribute assignment so that no
    real SentenceTransformer (torch) import occurs.  conftest.py already stubs
    the sentence_transformers module, but __new__ avoids the issue entirely.
    """

    def setUp(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss-cpu not installed")

        import faiss

        self._docs = _make_docs(5)
        self._embeddings = _make_embeddings(5)

        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)

        # Build and persist a real FAISS index (fast — no model loading)
        index = faiss.IndexFlatIP(_DIM)
        index.add(self._embeddings)
        self._faiss_path = tmp / "vector_index.faiss"
        self._meta_path = tmp / "vector_meta.json"
        faiss.write_index(index, str(self._faiss_path))
        self._meta_path.write_text(
            json.dumps(self._docs, ensure_ascii=True), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_backend(self):
        import faiss
        from treeqa.backends.vector import FaissVectorBackend

        # Construct without calling __init__ to avoid any library import cost
        backend = FaissVectorBackend.__new__(FaissVectorBackend)
        backend._index = faiss.read_index(str(self._faiss_path))
        backend.documents = self._docs
        backend._idf = backend._build_idf()
        backend._model = _fake_st_model(self._docs, self._embeddings)
        return backend

    def test_backend_loads_correct_document_count(self) -> None:
        backend = self._make_backend()
        self.assertEqual(len(backend.documents), 5)

    def test_semantic_search_self_query_ranks_first(self) -> None:
        """Querying with doc-2's own embedding should return doc-2 at rank 0."""
        from treeqa.backends.vector import FaissVectorBackend
        backend = self._make_backend()
        # Override encode to always return embedding of doc-2
        backend._model.encode.side_effect = lambda texts, **kw: self._embeddings[2:3]
        results = backend._semantic_search("any query", limit=5)
        self.assertEqual(results[0].source_id, "doc-2 [Main]")

    def test_search_returns_at_most_limit_results(self) -> None:
        backend = self._make_backend()
        results = backend.search("hybrid retrieval", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_search_source_type_is_vector(self) -> None:
        backend = self._make_backend()
        results = backend.search("film composer", limit=3)
        for r in results:
            self.assertEqual(r.source_type, "vector")

    def test_hybrid_no_duplicate_ids(self) -> None:
        backend = self._make_backend()
        results = backend._hybrid_search("Sruthilayalu Mahadevan", limit=5)
        ids = [r.source_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_lexical_search_rare_term_surfaces_correct_doc(self) -> None:
        backend = self._make_backend()
        results = backend._lexical_search("Sruthilayalu Telugu film", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0].source_id, "doc-0 [Main]")

    def test_idf_table_populated(self) -> None:
        backend = self._make_backend()
        self.assertGreater(len(backend._idf), 0)

    def test_faiss_index_dimensions_correct(self) -> None:
        import faiss
        backend = self._make_backend()
        self.assertEqual(backend._index.d, _DIM)
        self.assertEqual(backend._index.ntotal, 5)


# ---------------------------------------------------------------------------
# _build_faiss_index (ingest path)
# ---------------------------------------------------------------------------

class BuildFaissIndexTest(unittest.TestCase):
    def test_creates_faiss_and_meta_files(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss-cpu not installed")

        import faiss
        from treeqa.ingest import IndexedChunk, _build_faiss_index

        chunks = [
            IndexedChunk(
                source_id=f"c{i}",
                source_path="test.md",
                title="T",
                section="S",
                chunk_index=i,
                ingested_at="2026-01-01",
                content=f"chunk content number {i}",
            )
            for i in range(3)
        ]

        embeddings = _make_embeddings(3)
        with tempfile.TemporaryDirectory() as tmp:
            faiss_path = Path(tmp) / "idx.faiss"
            meta_path = Path(tmp) / "meta.json"

            with patch("sentence_transformers.SentenceTransformer") as MockST:
                instance = MockST.return_value
                instance.encode.return_value = embeddings
                _build_faiss_index(chunks, "all-MiniLM-L6-v2", faiss_path, meta_path)

            self.assertTrue(faiss_path.exists(), "FAISS index file not created")
            self.assertTrue(meta_path.exists(), "Metadata JSON not created")

            index = faiss.read_index(str(faiss_path))
            self.assertEqual(index.ntotal, 3)
            self.assertEqual(index.d, _DIM)

            meta = json.loads(meta_path.read_text())
            self.assertEqual(len(meta), 3)
            self.assertEqual(meta[0]["source_id"], "c0")

    def test_graceful_skip_when_faiss_missing(self) -> None:
        """_build_faiss_index should not raise when faiss is unavailable."""
        from treeqa.ingest import IndexedChunk, _build_faiss_index

        with patch.dict("sys.modules", {"faiss": None}):
            # Should return None silently
            result = _build_faiss_index([], "model", Path("/tmp/x.faiss"), Path("/tmp/x.json"))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Pipeline utilities — _enrich_query and _strip_sources
# ---------------------------------------------------------------------------

class PipelineUtilsTest(unittest.TestCase):
    def test_enrich_query_appends_prior_answers(self) -> None:
        from treeqa.pipeline import TreeQAPipeline

        enriched = TreeQAPipeline._enrich_query(
            "When was the composer born?",
            [("Who composed Sruthilayalu?", "K. V. Mahadevan")],
        )
        self.assertIn("K. V. Mahadevan", enriched)
        self.assertIn("When was the composer born?", enriched)

    def test_enrich_query_no_prior_hops_unchanged(self) -> None:
        from treeqa.pipeline import TreeQAPipeline

        q = "What is HotpotQA?"
        enriched = TreeQAPipeline._enrich_query(q, [])
        self.assertIn(q, enriched)

    def test_enrich_query_multiple_hops_all_appended(self) -> None:
        from treeqa.pipeline import TreeQAPipeline

        enriched = TreeQAPipeline._enrich_query(
            "Q3",
            [("Q1", "Answer one"), ("Q2", "Answer two")],
        )
        self.assertIn("Answer one", enriched)
        self.assertIn("Answer two", enriched)

    def test_strip_sources_removes_trailing_block(self) -> None:
        from treeqa.pipeline import TreeQAPipeline

        text = "K. V. Mahadevan was a composer. Sources: vector:doc-1, graph:fact-2"
        stripped = TreeQAPipeline._strip_sources(text)
        self.assertEqual(stripped, "K. V. Mahadevan was a composer.")

    def test_strip_sources_no_op_when_no_sources(self) -> None:
        from treeqa.pipeline import TreeQAPipeline

        text = "Plain answer with no sources block."
        self.assertEqual(TreeQAPipeline._strip_sources(text), text)


# ---------------------------------------------------------------------------
# Pipeline hop chaining — integration with mocks
# ---------------------------------------------------------------------------

class PipelineHopChainingTest(unittest.TestCase):
    """Verifies that prior hop answers propagate to retrieval and generation."""

    def _make_pipeline(self, generator=None, retriever=None):
        from unittest.mock import MagicMock
        from treeqa.agents.decomposer import QueryDecomposer
        from treeqa.config import TreeQASettings
        from treeqa.models import RetrievedDocument, ValidationResult
        from treeqa.pipeline import TreeQAPipeline

        settings = TreeQASettings(llm_provider="stub", vector_provider="memory", graph_provider="memory")

        if retriever is None:
            retriever = MagicMock()
            retriever.retrieve.return_value = [
                RetrievedDocument(source_id="d1", source_type="vector", content="Evidence.", score=0.9)
            ]

        if generator is None:
            generator = MagicMock()
            generator.generate_for_node.return_value = "Hop answer."
            generator.generate_final.return_value = "Hop answer."

        validator = MagicMock()
        validator.validate.return_value = ValidationResult(passed=True, confidence=0.9, rationale="OK")
        corrector = MagicMock()

        return TreeQAPipeline(
            settings=settings,
            decomposer=QueryDecomposer(llm_client=None),
            retriever=retriever,
            validator=validator,
            corrector=corrector,
            generator=generator,
        )

    def test_hop_answer_passed_as_prior_hop_to_generator(self) -> None:
        """Second sub-question generator call should receive prior_hops from hop 1."""
        call_args_list = []

        def capture_generate(question, documents, prior_hops=None):
            # Snapshot the list now — the parent _resolve_tree will mutate it after
            call_args_list.append((question, list(prior_hops) if prior_hops else []))
            return f"Answer for: {question}"

        from unittest.mock import MagicMock
        generator = MagicMock()
        generator.generate_for_node.side_effect = capture_generate
        generator.generate_final.return_value = "Final."

        pipeline = self._make_pipeline(generator=generator)
        # " and " causes rule-based decomposition into 2 sub-questions
        pipeline.run("What is HotpotQA and how does LangGraph work?")

        # Second call should have prior_hops from hop 1
        self.assertEqual(len(call_args_list), 2)
        _question1, hops1 = call_args_list[0]
        _question2, hops2 = call_args_list[1]
        # First hop has no prior hops
        self.assertFalse(hops1)
        # Second hop gets the first hop's answer
        self.assertEqual(len(hops2), 1)
        self.assertIn("What is HotpotQA", hops2[0][0])

    def test_retriever_query_enriched_with_prior_hop_answer(self) -> None:
        """Retriever must receive an enriched query on the second hop."""
        from unittest.mock import MagicMock
        from treeqa.models import RetrievedDocument

        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievedDocument(source_id="d1", source_type="vector", content="Evidence.", score=0.9)
        ]

        pipeline = self._make_pipeline(retriever=retriever)
        pipeline.run("What is HotpotQA and how does LangGraph work?")

        calls = [call.args[0] for call in retriever.retrieve.call_args_list]
        # Second retrieval call should include the first answer in the query string
        self.assertGreaterEqual(len(calls), 2)
        # First call is the plain sub-question; second is enriched
        self.assertNotEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
