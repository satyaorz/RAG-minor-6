from __future__ import annotations

from typing import Any

from treeqa.backends.llm import LLMClient
from treeqa.models import QueryNode


class QueryDecomposer:
    """Uses an LLM planner when configured, with a rule-based fallback."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def decompose(self, query: str) -> list[QueryNode]:
        llm_parts = self._decompose_with_llm(query)
        if llm_parts:
            return [
                QueryNode(node_id=f"node-{index}", question=part)
                for index, part in enumerate(llm_parts, start=1)
            ]
        parts = self._split_query(query)
        if len(parts) == 1:
            return [QueryNode(node_id="root", question=parts[0])]
        return [
            QueryNode(node_id=f"node-{index}", question=part)
            for index, part in enumerate(parts, start=1)
        ]

    def _split_query(self, query: str) -> list[str]:
        normalized = query.replace(" then ", " and ")
        segments = [segment.strip(" ?.") for segment in normalized.split(" and ")]
        cleaned = [segment for segment in segments if segment]
        return cleaned or [query.strip()]

    def _decompose_with_llm(self, query: str) -> list[str]:
        if self.llm_client is None:
            return []
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You decompose multi-hop questions into short, verifiable sub-questions. "
                    "Return only JSON in the form {\"sub_questions\": [\"...\"]}."
                ),
                user_prompt=f"Question: {query}",
            )
        except Exception:
            return []
        return self._extract_questions(payload)

    def _extract_questions(self, payload: dict[str, Any] | list[Any]) -> list[str]:
        if isinstance(payload, list):
            candidates = payload
        else:
            candidates = payload.get("sub_questions") or payload.get("questions") or []
        if not isinstance(candidates, list):
            return []
        return [str(candidate).strip() for candidate in candidates if str(candidate).strip()]
