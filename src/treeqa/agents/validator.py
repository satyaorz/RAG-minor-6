from __future__ import annotations

from treeqa.backends.llm import LLMClient
from treeqa.models import RetrievedDocument, ValidationResult

# Minimum number of retrieved documents required to consider an answer verified.
_MIN_EVIDENCE_COUNT = 2
# Minimum lexical overlap ratio for heuristic validation.
_MIN_OVERLAP_RATIO = 0.35


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

        if len(documents) < _MIN_EVIDENCE_COUNT:
            return ValidationResult(
                passed=False,
                confidence=0.1,
                rationale=(
                    f"Only {len(documents)} document(s) retrieved; "
                    f"at least {_MIN_EVIDENCE_COUNT} required to verify an answer."
                ),
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
        passed = confidence >= _MIN_OVERLAP_RATIO
        rationale = (
            "Answer is grounded in retrieved evidence (lexical overlap)."
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
                    "You are a strict factuality judge. "
                    "Evaluate whether the answer is directly supported by the supplied evidence. "
                    "Return ONLY a JSON object with exactly three keys: "
                    "\"passed\" (boolean — true only when all key claims are clearly supported), "
                    "\"confidence\" (number from 0.0 to 1.0), "
                    "\"rationale\" (one concise sentence explaining the verdict). "
                    "Do not add any other keys or text outside the JSON object."
                ),
                user_prompt=(
                    f"Answer:\n{answer}\n\n"
                    f"Evidence:\n{self._format_context(documents)}\n\n"
                    "Is every factual claim in the answer directly supported by the evidence above?"
                ),
            )
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        passed = bool(payload.get("passed", False))
        raw_confidence = payload.get("confidence", 0.0)
        confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else 0.0
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(payload.get("rationale", "")).strip()
        if not rationale:
            rationale = "LLM judge returned no rationale."
        return ValidationResult(passed=passed, confidence=confidence, rationale=rationale)

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        return "\n".join(
            f"[{document.source_type}:{document.source_id}] {document.content}"
            for document in documents
        )

