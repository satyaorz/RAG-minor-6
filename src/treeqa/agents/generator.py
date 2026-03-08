from __future__ import annotations

from treeqa.backends.llm import LLMClient
from treeqa.models import QueryNode, RetrievedDocument


class AnswerGenerator:
    """Grounded answer composer with optional LLM-backed synthesis."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate_for_node(self, question: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return f"Insufficient evidence to answer: {question}"
        if self.llm_client is not None:
            try:
                context = self._format_context(documents)
                answer = self.llm_client.generate_text(
                    system_prompt=(
                        "Answer using only the supplied evidence. "
                        "If the evidence is insufficient, say so explicitly."
                    ),
                    user_prompt=f"Question: {question}\n\nEvidence:\n{context}",
                )
                if answer:
                    return answer
            except Exception:
                pass
        best_document = max(documents, key=lambda document: document.score)
        return best_document.content

    def generate_final(self, query: str, nodes: list[QueryNode]) -> str:
        grounded_answers = [node.answer for node in nodes if node.answer]
        if not grounded_answers:
            return f"No grounded answer could be produced for: {query}"
        if self.llm_client is not None:
            try:
                outline = "\n".join(
                    f"- {node.question}: {node.answer}" for node in nodes if node.answer
                )
                answer = self.llm_client.generate_text(
                    system_prompt=(
                        "Combine verified sub-answers into a concise final response. "
                        "Do not introduce new facts."
                    ),
                    user_prompt=f"Original query: {query}\n\nVerified notes:\n{outline}",
                )
                if answer:
                    return answer
            except Exception:
                pass
        return " ".join(grounded_answers)

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        return "\n".join(
            f"[{document.source_type}:{document.source_id}] {document.content}"
            for document in documents
        )
