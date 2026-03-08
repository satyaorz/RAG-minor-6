from __future__ import annotations

import re

from treeqa.backends.llm import LLMClient
from treeqa.models import QueryNode, RetrievedDocument
from treeqa.retrieval.scoring import select_relevant_snippet


class AnswerGenerator:
    """Grounded answer composer with optional LLM-backed synthesis."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate_for_node(self, question: str, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return f"Insufficient evidence to answer: {question}"
        selected_documents = documents[:3]
        if self.llm_client is not None:
            try:
                context = self._format_context(question, selected_documents)
                answer = self.llm_client.generate_text(
                    system_prompt=(
                        "Answer using only the supplied evidence. "
                        "Write 2 to 4 sentences in plain text. "
                        "Prefer synthesis over copying. "
                        "Do not repeat document titles or headings. "
                        "Keep the answer concise, factual, and free of markdown lists unless the question asks for a list. "
                        "If the evidence is insufficient, say so explicitly. "
                        "End with a short `Sources:` line using the supplied source ids."
                    ),
                    user_prompt=(
                        f"Question: {question}\n\n"
                        f"Evidence:\n{context}\n\n"
                        f"Use only these sources: {self._source_refs(selected_documents)}"
                    ),
                )
                if answer:
                    return self._clean_text(answer)
            except Exception:
                pass
        best_document = max(selected_documents, key=lambda document: document.score)
        snippet = select_relevant_snippet(best_document.content, question)
        return self._clean_text(
            f"{snippet} Sources: {self._source_refs([best_document])}"
        )

    def generate_final(self, query: str, nodes: list[QueryNode]) -> str:
        grounded_answers = [self._strip_sources(node.answer) for node in nodes if node.answer]
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
                        "Do not introduce new facts. "
                        "Write a short paragraph or a short intro plus a compact list when useful. "
                        "Do not copy the sub-answers verbatim. "
                        "End with a `Sources:` line."
                    ),
                    user_prompt=(
                        f"Original query: {query}\n\n"
                        f"Verified notes:\n{outline}\n\n"
                        f"Available sources: {self._node_source_refs(nodes)}"
                    ),
                )
                if answer:
                    return self._clean_text(answer)
            except Exception:
                pass
        combined = " ".join(self._dedupe_answers(grounded_answers))
        return self._clean_text(f"{combined} Sources: {self._node_source_refs(nodes)}")

    def _format_context(self, question: str, documents: list[RetrievedDocument]) -> str:
        return "\n".join(
            f"[{document.source_type}:{document.source_id}] {select_relevant_snippet(document.content, question)}"
            for document in documents
        )

    def _source_refs(self, documents: list[RetrievedDocument]) -> str:
        refs: list[str] = []
        seen: set[str] = set()
        for document in documents:
            ref = f"{document.source_type}:{document.source_id}"
            if ref in seen:
                continue
            refs.append(ref)
            seen.add(ref)
        return ", ".join(refs[:5])

    def _node_source_refs(self, nodes: list[QueryNode]) -> str:
        refs: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            for document in node.documents:
                ref = f"{document.source_type}:{document.source_id}"
                if ref in seen:
                    continue
                refs.append(ref)
                seen.add(ref)
        return ", ".join(refs[:6])

    def _dedupe_answers(self, answers: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for answer in answers:
            normalized = self._clean_text(self._strip_sources(answer))
            if normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
        return cleaned

    def _clean_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
        compact = compact.replace("Sources :", "Sources:")
        compact = re.sub(r"(Sources:\s*[^.]+)\s+Sources:\s*", r"\1, ", compact)
        compact = compact.replace("# ", "")
        return compact

    def _strip_sources(self, text: str) -> str:
        return re.sub(r"\s*Sources:\s.*$", "", text, flags=re.IGNORECASE).strip()
