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
    def __init__(self, index_path: str) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise RuntimeError(
                f"Local vector index not found at {self.index_path}. Run `python -m treeqa.cli ingest`."
            )
        self.documents = self._load_index()

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        question_terms = {term.lower().strip("?,.") for term in question.split() if term}
        results: list[RetrievedDocument] = []
        for record in self.documents:
            normalized = normalize_text(str(record["content"]))
            score = lexical_score(question, normalized)
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(
                    source_id=str(record["source_id"]),
                    source_type="vector",
                    content=normalized,
                    score=score,
                )
            )
        return sorted(results, key=lambda document: document.score, reverse=True)[:limit]

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
        return LocalVectorBackend(index_path=str(index_path))
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
