from __future__ import annotations

from hamhrag.backends.llm import LLMClient


class CorrectionEngine:
    """Produces refined queries when validation fails."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def refine(self, question: str, attempt: int) -> str:
        if self.llm_client is not None:
            try:
                rewrite = self.llm_client.generate_text(
                    system_prompt=(
                        "Rewrite the question for retrieval. Keep it short and factual. "
                        "Return only the rewritten query."
                    ),
                    user_prompt=f"Attempt: {attempt}\nQuestion: {question}",
                )
                if rewrite:
                    return rewrite.strip()
            except Exception:
                pass
        cleaned = question.strip()
        if not cleaned:
            return question
        if not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        lowered = cleaned.lower()
        if "director" in lowered and ("country" in lowered or "nationality" in lowered):
            # Keep retries entity-focused instead of drifting toward framework terms.
            return f"{cleaned} director nationality country of origin"
        return cleaned

    def generate_variants(self, question: str, num_variants: int = 3) -> list[str]:
        """Generate diverse semantic variants of the query for parallel speculative execution."""
        if self.llm_client is not None:
            try:
                text = self.llm_client.generate_text(
                    system_prompt=(
                        "You are an expert at information retrieval. "
                        f"Rewrite the given question into {num_variants} diverse semantic variations "
                        "that can be used to query a vector database. Use different phrasing, synonyms, "
                        "and entity-focused keywords for each variant. Make sure they all aim to answer the original question. "
                        "Do not include placeholder words like <person>.\n"
                        "Return EXACTLY one variant per line. Do not include numbers, bullets, or any other text."
                    ),
                    user_prompt=f"Question: {question}",
                )
                
                if text:
                    variants = [line.strip("-*0123456789. \t\"'") for line in text.strip().split("\n")]
                    # Filter out empty, very short, and exact matches
                    cleaned = [v for v in variants if v and len(v) > 5 and v.lower() != question.lower()]
                    if cleaned:
                        return cleaned[:num_variants]
            except Exception:
                pass
        
        # Fallback if LLM fails
        return []
