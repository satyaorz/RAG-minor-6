from __future__ import annotations

from treeqa.backends.llm import LLMClient
from treeqa.models import RetrievedDocument, ValidationResult


class AnswerValidator:
    """Uses an LLM judge when configured, otherwise falls back to heuristics."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def validate(self, answer: str, documents: list[RetrievedDocument]) -> ValidationResult:
        if not documents:
            return ValidationResult(
                passed=False,
                confidence=0.0,
                rationale="No evidence was retrieved for this node.",
            )
        if self.llm_client is not None:
            llm_result = self._validate_with_llm(answer, documents)
            if llm_result is not None:
                return llm_result

        answer_terms = {term.lower() for term in answer.split() if term}
        evidence_terms = {
            term.lower()
            for document in documents
            for term in document.content.split()
            if term
        }
        overlap = len(answer_terms & evidence_terms)
        confidence = min(1.0, overlap / max(len(answer_terms), 1))
        passed = confidence >= 0.3
        rationale = (
            "Answer is partially grounded in retrieved evidence."
            if passed
            else "Evidence support is insufficient; retry recommended."
        )
        return ValidationResult(passed=passed, confidence=confidence, rationale=rationale)

    def _validate_with_llm(
        self, answer: str, documents: list[RetrievedDocument]
    ) -> ValidationResult | None:
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a factuality judge. Return only JSON with keys "
                    "\"passed\" (boolean), \"confidence\" (0 to 1), and \"rationale\" (string)."
                ),
                user_prompt=(
                    f"Answer:\n{answer}\n\nEvidence:\n{self._format_context(documents)}"
                ),
            )
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        passed = bool(payload.get("passed", False))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        rationale = str(payload.get("rationale", ""))
        return ValidationResult(passed=passed, confidence=confidence, rationale=rationale)

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        return "\n".join(
            f"[{document.source_type}:{document.source_id}] {document.content}"
            for document in documents
        )
