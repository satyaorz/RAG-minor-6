from __future__ import annotations

from dataclasses import dataclass

from treeqa.backends.llm import LLMClient
from treeqa.models import RetrievedDocument


@dataclass(slots=True)
class ConflictResolution:
    query: str
    reason: str


class ConflictAuditor:
    """Resolves vector vs graph evidence conflicts by proposing a focused follow-up query."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def resolve(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> ConflictResolution | None:
        if self.llm_client is None:
            return None
        if not documents:
            return None
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a conflict auditor for a multi-hop RAG system. "
                    "Vector and graph evidence disagree. "
                    "Propose ONE precise follow-up retrieval query that will disambiguate the conflict. "
                    "If the conflict cannot be resolved by retrieval, return a query that asks for the missing fact. "
                    "Return JSON: {\"query\": string, \"reason\": string}."
                ),
                user_prompt=(
                    f"Question: {question}\n"
                    f"Draft Answer: {answer}\n"
                    f"Evidence:\n{self._format_context(documents)}\n"
                ),
            )
            if not isinstance(payload, dict):
                return None
            query = str(payload.get("query", "")).strip()
            reason = str(payload.get("reason", "")).strip()
            if not query:
                return None
            return ConflictResolution(query=query, reason=reason)
        except Exception:
            return None

    @staticmethod
    def _format_context(documents: list[RetrievedDocument]) -> str:
        return "\n".join(f"[{d.source_type}] {d.content}" for d in documents)
