from __future__ import annotations

import re

from hamhrag.backends.llm import LLMClient
from hamhrag.models import RetrievedDocument, ValidationResult
from hamhrag.retrieval.scoring import normalize_text, tokenize

# Minimum number of retrieved documents required to consider an answer verified.
_MIN_EVIDENCE_COUNT = 2
# Minimum lexical overlap ratio for heuristic validation.
_MIN_OVERLAP_RATIO = 0.35
_SINGLE_EVIDENCE_SCORE = 0.75
_CONFLICT_THRESHOLD = 0.45


class AnswerValidator:
    """Uses a consolidated LLM judge to verify grounding, consensus, and category match."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def validate(
        self,
        answer: str,
        documents: list[RetrievedDocument],
        question: str = "",
    ) -> ValidationResult:
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

        # Detect any form of "I don't know" / "evidence doesn't say" answers.
        # These must NEVER be marked as verified — they contain no factual payload.
        _NO_EVIDENCE_PHRASES = (
            "insufficient evidence",
            "no grounded answer",
            "does not mention",
            "does not explicitly mention",
            "does not provide",
            "does not contain",
            "does not include",
            "does not state",
            "does not specify",
            "no information",
            "no evidence",
            "not mentioned",
            "not explicitly mentioned",
            "not provided",
            "cannot be determined",
            "cannot determine",
            "not found in",
            "not in the evidence",
            "no mention",
            "evidence provided does not",
            "provided evidence does not",
            "sources do not",
            "evidence does not",
        )
        if any(phrase in lowered for phrase in _NO_EVIDENCE_PHRASES):
            return ValidationResult(
                passed=False,
                confidence=0.0,
                rationale="Answer indicates insufficient evidence — the retrieval did not find the relevant document.",
            )

        consensus_score = self._consensus_score(documents)
        hard_alignment = self._question_alignment(question, answer, documents)
        if not hard_alignment:
            return ValidationResult(
                passed=False,
                confidence=0.1,
                rationale="Answer is supported but does not answer the requested category/context.",
                category_match=False,
                source_conflict=consensus_score < _CONFLICT_THRESHOLD,
                consensus_score=consensus_score,
            )

        confidence = self._evidence_overlap_confidence(answer, documents)
        conflict = consensus_score < _CONFLICT_THRESHOLD
        if confidence >= 0.65 and not conflict:
            return ValidationResult(
                passed=True,
                confidence=confidence,
                rationale="High-confidence lexical validation.",
                category_match=True,
                source_conflict=False,
                consensus_score=consensus_score,
            )

        if self.llm_client is not None:
            return self._validate_consolidated(
                answer,
                documents,
                consensus_score,
                question=question,
            )

        # Heuristic Fallback
        return ValidationResult(
            passed=confidence >= _MIN_OVERLAP_RATIO and not conflict,
            confidence=confidence,
            rationale="Lexical overlap validation.",
            category_match=True,
            source_conflict=conflict,
            consensus_score=consensus_score,
        )

    def _validate_consolidated(
        self,
        answer: str,
        documents: list[RetrievedDocument],
        consensus_score: float,
        question: str = "",
    ) -> ValidationResult:
        """Single LLM call to handle Grounding, Consensus, and Category Verification."""
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a logic auditor. Evaluate the provided answer against the evidence.\n"
                    "CRITICAL RULES:\n"
                    "1. Entity Alignment: If the Question specifies an entity (e.g. 'Theodore Roosevelt'), the answer MUST be about THAT entity. Reject (passed=false) if the answer or evidence refers to a different entity (e.g. 'George VI').\n"
                    "2. Grounding: Is the factual claim (dates, names, places) explicitly supported by the provided evidence text?\n"
                    "3. Category: Does it answer the Question's requested type (e.g. a date for 'when', a person for 'who')?\n"
                    "4. Consensus: If sources contradict each other, mark source_conflict=true.\n\n"
                    "Return JSON: {\"passed\": bool, \"confidence\": float, \"rationale\": string, \"category_match\": bool, \"source_conflict\": bool, \"consensus_score\": float}"
                ),
                user_prompt=(
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Evidence: {self._format_context(documents)}\n"
                ),
            )
            if not isinstance(payload, dict):
                return ValidationResult(passed=False, confidence=0.0, rationale="Judge failure.")
                
            cat_match = bool(payload.get("category_match", True))
            conflict = bool(payload.get("source_conflict", False))
            consensus = float(payload.get("consensus_score", consensus_score))
            if consensus < _CONFLICT_THRESHOLD:
                conflict = True
            passed = bool(payload.get("passed", False)) and cat_match and not conflict
            
            conf = float(payload.get("confidence", 0.0))
            if not cat_match or conflict:
                conf = min(conf, 0.3)
                
            rat = str(payload.get("rationale", ""))
            if not cat_match: rat = f"[Category Mismatch] {rat}"
            if conflict: rat = f"[Source Conflict] {rat}"
                
            return ValidationResult(
                passed=passed,
                confidence=conf,
                rationale=rat,
                category_match=cat_match,
                source_conflict=conflict,
                consensus_score=consensus,
            )
        except Exception:
            return ValidationResult(passed=False, confidence=0.0, rationale="Validation error.")

    def _question_alignment(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> bool:
        lowered_q = question.lower()
        lowered_a = answer.lower()
        if not lowered_q:
            return True

        # --- Entity Grounding check ---
        # If the question explicitly names a proper-noun entity (e.g. "Theodore Roosevelt"),
        # that entity MUST appear in at least one document and the answer.
        # This prevents "George VI" hallucinations when searching for "Theodore Roosevelt".
        named_entities = re.findall(r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+)+\b", question)
        if named_entities:
            for entity in named_entities:
                entity_lower = entity.lower()
                # Check if entity appears in evidence
                in_evidence = any(entity_lower in doc.content.lower() for doc in documents)
                if not in_evidence:
                    return False
                # Check if entity (or part of it) appears in answer
                # (Relaxed check: just needs to mention the surname or full name)
                surname = entity.split()[-1].lower()
                if surname not in lowered_a:
                    return False

        # --- Birthday / birth date check ---
        # "When was X born?" or "What is X's birthday?" → must contain a date.
        if (re.search(r"\bborn\b", lowered_q) or "birthday" in lowered_q) and "when" in lowered_q:
            has_date = bool(re.search(
                r"\b(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4})\b",
                lowered_a,
            ))
            if not has_date:
                return False
            
            # Reject if answer covers multiple people's birthdays without matching the subject.
            multi_person_dates = re.findall(
                r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s+(?:was born|born on|born in|birthday is)",
                answer,
            )
            if len(multi_person_dates) > 1:
                name_match = re.search(
                    r"(?:when was|birthday of|birthday is)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
                    question,
                    re.IGNORECASE,
                )
                if name_match:
                    subject = name_match.group(1).lower()
                    if not any(subject in n.lower() for n in multi_person_dates):
                        return False
                else:
                    return False
            return True

        if "when" in lowered_q and re.search(
            r"\b(created|founded|formed|established|started|inaugurated)\b",
            lowered_q,
        ):
            has_creation_language = re.search(
                r"\b(created|founded|formed|established|started|inaugurated|creation|founding|establishment)\b",
                lowered_a,
            )
            has_year = re.search(r"\b(?:1[5-9]\d{2}|20\d{2})\b", lowered_a)
            return bool(has_creation_language and has_year)

        if "body of water" in lowered_q:
            return bool(
                re.search(
                    r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
                    r"(?:Sea|Ocean|Bay|Strait|Reservoir|Lake|River)\b",
                    answer,
                )
            )

        water = re.search(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
            r"(?:Sea|Ocean|Bay|Strait|Reservoir|Lake|River))\b",
            question,
        )
        if water and re.search(r"\b(borders?|shore|coast|adjacent|located)\b", lowered_q):
            body = water.group(1).lower()
            return any(body in document.content.lower() for document in documents) or body in lowered_a

        return True


    @staticmethod
    def _evidence_overlap_confidence(
        answer: str,
        documents: list[RetrievedDocument],
    ) -> float:
        answer_body = re.sub(r"\s*Sources:\s.*$", "", answer, flags=re.IGNORECASE)
        answer_tokens = set(tokenize(normalize_text(answer_body)))
        if not answer_tokens:
            return 0.0
        evidence_tokens = {
            token
            for document in documents
            for token in tokenize(normalize_text(document.content))
        }
        if not evidence_tokens:
            return 0.0
        return min(1.0, len(answer_tokens & evidence_tokens) / len(answer_tokens))

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        return "\n".join(f"[{d.source_type}] {d.content}" for d in documents)

    def _consensus_score(self, documents: list[RetrievedDocument]) -> float:
        sources: dict[str, list[RetrievedDocument]] = {}
        for document in documents:
            sources.setdefault(document.source_type, []).append(document)

        if len(sources) <= 1:
            return 1.0

        vector_docs = sources.get("vector") or []
        graph_docs = sources.get("graph") or []
        if not vector_docs or not graph_docs:
            return 0.6

        top_vector = max(vector_docs, key=lambda doc: doc.score)
        top_graph = max(graph_docs, key=lambda doc: doc.score)

        vec_tokens = set(tokenize(normalize_text(top_vector.content)))
        graph_tokens = set(tokenize(normalize_text(top_graph.content)))
        if not vec_tokens or not graph_tokens:
            return 0.0

        overlap = len(vec_tokens & graph_tokens)
        union = len(vec_tokens | graph_tokens)
        return overlap / union if union else 0.0
