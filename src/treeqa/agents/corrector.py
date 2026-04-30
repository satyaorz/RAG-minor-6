from __future__ import annotations

from treeqa.backends.llm import LLMClient


class CorrectionEngine:
    """Produces a refined query when validation fails."""

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
