from __future__ import annotations

from treeqa.backends.llm import LLMClient
from treeqa.models import RetrievedDocument, ValidationResult

# Minimum number of retrieved documents required to consider an answer verified.
_MIN_EVIDENCE_COUNT = 2
# Minimum lexical overlap ratio for heuristic validation.
_MIN_OVERLAP_RATIO = 0.35
_SINGLE_EVIDENCE_SCORE = 0.75


class AnswerValidator:
    """Uses a consolidated LLM judge to verify grounding, consensus, and category match."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def validate(self, answer: str, documents: list[RetrievedDocument]) -> ValidationResult:
        if not documents:
            return ValidationResult(passed=False, confidence=0.0, rationale="No evidence found.")

        allow_single = (
            len(documents) == 1
            and (documents[0].score >= _SINGLE_EVIDENCE_SCORE or documents[0].source_type == "graph")
        )
        if len(documents) < _MIN_EVIDENCE_COUNT and not allow_single:
            return ValidationResult(
                passed=False, 
                confidence=0.1, 
                rationale=f"Insufficient evidence count ({len(documents)})."
            )

        lowered = answer.strip().lower()
        if lowered.startswith("insufficient evidence") or lowered.startswith("no grounded answer"):
            return ValidationResult(
                passed=False,
                confidence=0.0,
                rationale="Answer indicates insufficient evidence.",
            )

        if self.llm_client is not None:
            return self._validate_consolidated(answer, documents)

        # Heuristic Fallback
        answer_terms = {term.lower() for term in answer.split() if term}
        evidence_terms = {term.lower() for d in documents for term in d.content.split() if term}
        overlap = len(answer_terms & evidence_terms)
        confidence = min(1.0, overlap / max(len(answer_terms), 1))
        return ValidationResult(
            passed=confidence >= _MIN_OVERLAP_RATIO, 
            confidence=confidence, 
            rationale="Lexical overlap validation."
        )

    def _validate_consolidated(self, answer: str, documents: list[RetrievedDocument]) -> ValidationResult:
        """Single LLM call to handle Grounding, Consensus, and Category Verification."""
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a logic auditor. Evaluate the provided answer against the evidence.\n"
                    "Check for:\n"
                    "1. Grounding: Is it supported by evidence?\n"
                    "2. Category: Does it provide the specific type requested (e.g. Country vs Region)?\n"
                    "3. Consensus: Do the sources agree (Vector vs Graph)?\n\n"
                    "Return JSON: {\"passed\": bool, \"confidence\": float, \"rationale\": string, \"category_match\": bool, \"source_conflict\": bool}"
                ),
                user_prompt=(
                    f"Answer: {answer}\n"
                    f"Evidence: {self._format_context(documents)}\n"
                ),
            )
            if not isinstance(payload, dict):
                return ValidationResult(passed=False, confidence=0.0, rationale="Judge failure.")
                
            cat_match = bool(payload.get("category_match", True))
            conflict = bool(payload.get("source_conflict", False))
            passed = bool(payload.get("passed", False)) and cat_match and not conflict
            
            conf = float(payload.get("confidence", 0.0))
            if not cat_match or conflict:
                conf = min(conf, 0.3)
                
            rat = str(payload.get("rationale", ""))
            if not cat_match: rat = f"[Category Mismatch] {rat}"
            if conflict: rat = f"[Source Conflict] {rat}"
                
            return ValidationResult(passed=passed, confidence=conf, rationale=rat)
        except Exception:
            return ValidationResult(passed=False, confidence=0.0, rationale="Validation error.")

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        return "\n".join(f"[{d.source_type}] {d.content}" for d in documents)
