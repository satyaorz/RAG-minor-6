from __future__ import annotations

from typing import Any, Protocol

from treeqa.config import TreeQASettings
from treeqa.models import RetrievedDocument


class GraphBackend(Protocol):
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        ...


class MemoryGraphBackend:
    def __init__(self, facts: dict[str, str] | None = None) -> None:
        self.facts = facts or {
            "fact-1": "TreeQA uses logic-tree reasoning to verify intermediate steps.",
            "fact-2": "Hybrid retrieval combines vector search with graph lookups.",
        }

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        question_terms = {term.lower().strip("?,.") for term in question.split() if term}
        results: list[RetrievedDocument] = []
        for source_id, content in self.facts.items():
            content_terms = {term.lower().strip("?,.") for term in content.split() if term}
            overlap = len(question_terms & content_terms)
            if overlap == 0:
                continue
            results.append(
                RetrievedDocument(
                    source_id=source_id,
                    source_type="graph",
                    content=content,
                    score=overlap / max(len(question_terms), 1),
                )
            )
        return sorted(results, key=lambda document: document.score, reverse=True)[:limit]


class Neo4jGraphBackend:
    def __init__(
        self,
        uri: str,
        database: str,
        username: str,
        password: str,
        index_name: str,
        query: str,
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError("neo4j is required for TREEQA_GRAPH_PROVIDER=neo4j.") from error

        auth = (username, password) if username else None
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database or None
        self.index_name = index_name
        self.query = query

    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        params = {"question": question, "limit": limit, "index_name": self.index_name}
        with self.driver.session(database=self.database) as session:
            records = session.run(self.query, **params)
            return [self._to_document(record) for record in records]

    def _to_document(self, record: Any) -> RetrievedDocument:
        source_id = record.get("source_id", "")
        content = record.get("content", "")
        score = float(record.get("score", 0.0) or 0.0)
        return RetrievedDocument(
            source_id=str(source_id),
            source_type="graph",
            content=str(content),
            score=score,
        )


def build_graph_backend(settings: TreeQASettings) -> GraphBackend:
    provider = settings.graph_provider.strip().lower()
    if provider in {"", "memory"}:
        return MemoryGraphBackend()
    if provider == "neo4j":
        if not settings.graph_store_url:
            raise ValueError("GRAPH_STORE_URL must be set when TREEQA_GRAPH_PROVIDER=neo4j.")
        return Neo4jGraphBackend(
            uri=settings.graph_store_url,
            database=settings.graph_database,
            username=settings.graph_username,
            password=settings.graph_password,
            index_name=settings.graph_index_name,
            query=settings.graph_query,
        )
    raise ValueError(f"Unsupported graph provider: {settings.graph_provider}")
