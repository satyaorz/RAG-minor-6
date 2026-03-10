from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from treeqa.config import TreeQASettings
from treeqa.models import RetrievedDocument
from treeqa.retrieval.scoring import lexical_score, normalize_text


class VectorBackend(Protocol):
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        ...


class MemoryVectorBackend:
    def __init__(self, documents: dict[str, str] | None = None) -> None:
        self.documents = documents or {
            "doc-1": "HotpotQA is a benchmark for multi-hop question answering.",
            "doc-2": "LangGraph helps orchestrate multi-step agent workflows.",
            "doc-3": "Wikidata and Neo4j can provide structured factual evidence.",
        }

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        results: list[RetrievedDocument] = []
        for source_id, content in self.documents.items():
            normalized = normalize_text(content)
            score = lexical_score(question, normalized)
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(
                    source_id=source_id,
                    source_type="vector",
                    content=normalized,
                    score=score,
                )
            )
        return sorted(results, key=lambda document: document.score, reverse=True)[:limit]


class QdrantVectorBackend:
    def __init__(self, url: str, collection_name: str, api_key: str = "") -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError(
                "qdrant-client is required for TREEQA_VECTOR_PROVIDER=qdrant."
            ) from error

        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection_name = collection_name

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=question,
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        results: list[RetrievedDocument] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            content = payload.get("content") or payload.get("text") or json.dumps(payload)
            results.append(
                RetrievedDocument(
                    source_id=str(payload.get("id") or getattr(point, "id", "")),
                    source_type="vector",
                    content=content,
                    score=float(getattr(point, "score", 0.0) or 0.0),
                )
            )
        return results


class LocalVectorBackend:
    """Embedding-based retrieval backend backed by a local JSONL index.

    At construction time the index is loaded and every document is encoded
    once with a sentence-transformer model.  Queries are answered with cosine
    similarity (dot-product on L2-normalised vectors).  If sentence-transformers
    is not installed the backend silently falls back to keyword scoring.
    """

    def __init__(self, index_path: str, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise RuntimeError(
                f"Local vector index not found at {self.index_path}. Run `python -m treeqa.cli ingest`."
            )
        self.documents = self._load_index()
        self._model, self._doc_embeddings = self._build_embeddings(embedding_model)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self, model_name: str):
        """Return (model, embeddings_array) or (None, None) on import error."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            return None, None

        model = SentenceTransformer(model_name)
        texts = [self._scoring_text(r) for r in self.documents]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return model, embeddings

    @staticmethod
    def _scoring_text(record: dict) -> str:
        title = str(record.get("title", "")).strip()
        section = str(record.get("section", "")).strip()
        content = normalize_text(str(record.get("content", "")))
        return " ".join(filter(None, [title, section, content]))

    @staticmethod
    def _source_id(record: dict) -> str:
        sid = str(record.get("source_id", ""))
        section = str(record.get("section", "")).strip()
        return f"{sid} [{section}]" if section else sid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        if self._model is not None and self._doc_embeddings is not None:
            return self._semantic_search(question, limit)
        return self._lexical_search(question, limit)

    # ------------------------------------------------------------------
    # Search implementations
    # ------------------------------------------------------------------

    def _semantic_search(self, question: str, limit: int) -> list[RetrievedDocument]:
        import numpy as np  # guaranteed present when sentence-transformers is installed

        q_emb = self._model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        # cosine similarity = dot product of L2-normalised vectors
        scores = (self._doc_embeddings @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:limit]

        results: list[RetrievedDocument] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            record = self.documents[idx]
            results.append(
                RetrievedDocument(
                    source_id=self._source_id(record),
                    source_type="vector",
                    content=normalize_text(str(record.get("content", ""))),
                    score=score,
                )
            )
        return results

    def _lexical_search(self, question: str, limit: int) -> list[RetrievedDocument]:
        results: list[RetrievedDocument] = []
        for record in self.documents:
            content = normalize_text(str(record.get("content", "")))
            score = lexical_score(question, self._scoring_text(record))
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(
                    source_id=self._source_id(record),
                    source_type="vector",
                    content=content,
                    score=score,
                )
            )
        return sorted(results, key=lambda d: d.score, reverse=True)[:limit]

    def _load_index(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


def build_vector_backend(settings: TreeQASettings) -> VectorBackend:
    provider = settings.vector_provider.strip().lower()
    if provider in {"", "memory"}:
        return MemoryVectorBackend()
    if provider == "local":
        index_path = (
            settings.resolve_path(settings.local_vector_index_path)
            if settings.local_vector_index_path
            else settings.resolved_data_dir / "index" / "vector_index.jsonl"
        )
        return LocalVectorBackend(
            index_path=str(index_path),
            embedding_model=settings.embedding_model,
        )
    if provider == "qdrant":
        if not settings.vector_store_url:
            raise ValueError("VECTOR_STORE_URL must be set when TREEQA_VECTOR_PROVIDER=qdrant.")
        if not settings.vector_collection:
            raise ValueError(
                "TREEQA_VECTOR_COLLECTION must be set when TREEQA_VECTOR_PROVIDER=qdrant."
            )
        return QdrantVectorBackend(
            url=settings.vector_store_url,
            collection_name=settings.vector_collection,
            api_key=settings.vector_api_key,
        )
    raise ValueError(f"Unsupported vector provider: {settings.vector_provider}")
