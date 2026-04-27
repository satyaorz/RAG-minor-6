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

        # Calculate Source Consensus Coefficient (SCC)
        source_types = {doc.source_type for doc in documents}
        consensus_score = 1.0
        if len(source_types) > 1:
            consensus_score = self._calculate_consensus(documents)

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
            llm_result = self._validate_with_llm(answer, documents, consensus_score)
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

    def _calculate_consensus(self, documents: list[RetrievedDocument]) -> float:
        """Determines if different source types (vector vs graph) contradict each other."""
        if not self.llm_client:
            return 1.0
            
        vector_docs = [d for d in documents if d.source_type == "vector"]
        graph_docs = [d for d in documents if d.source_type == "graph"]
        
        if not vector_docs or not graph_docs:
            return 1.0
            
        try:
            prompt = (
                "Compare the following two sets of evidence. "
                "Set A (Vector/Unstructured) vs Set B (Graph/Structured).\n\n"
                f"Set A: {self._format_context(vector_docs)}\n\n"
                f"Set B: {self._format_context(graph_docs)}\n\n"
                "Do these sources contradict each other regarding the core facts? "
                "Return a score from 0.0 (total contradiction) to 1.0 (perfect agreement). "
                "Return ONLY the number."
            )
            response = self.llm_client.generate_text(system_prompt="You are a consistency auditor.", user_prompt=prompt)
            return float(response.strip())
        except Exception:
            return 1.0

    def _validate_with_llm(
        self, answer: str, documents: list[RetrievedDocument], consensus_score: float = 1.0
    ) -> ValidationResult | None:
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a strict factuality judge. "
                    "Evaluate whether the answer is directly supported by the supplied evidence. "
                    "Consider the Source Consensus Coefficient (SCC) provided; lower SCC means sources disagree. "
                    "Return ONLY a JSON object with exactly three keys: "
                    "\"passed\" (boolean), \"confidence\" (number), \"rationale\" (string)."
                ),
                user_prompt=(
                    f"Answer: {answer}\n"
                    f"Evidence: {self._format_context(documents)}\n"
                    f"Source Consensus Coefficient: {consensus_score}\n\n"
                    "Is the answer fully supported?"
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

