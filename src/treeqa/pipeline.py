from __future__ import annotations

from treeqa.agents import (
    AnswerGenerator,
    AnswerValidator,
    CorrectionEngine,
    QueryDecomposer,
)
from treeqa.backends import build_graph_backend, build_llm_client, build_vector_backend
from treeqa.config import TreeQASettings
from treeqa.models import PipelineResult, QueryNode, ValidationResult
from treeqa.retrieval import HybridRetriever
from treeqa.state import WorkflowState


class TreeQAPipeline:
    def __init__(
        self,
        settings: TreeQASettings | None = None,
        decomposer: QueryDecomposer | None = None,
        retriever: HybridRetriever | None = None,
        validator: AnswerValidator | None = None,
        corrector: CorrectionEngine | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or TreeQASettings.from_env()
        llm_client = build_llm_client(self.settings)
        self.decomposer = decomposer or QueryDecomposer(llm_client=llm_client)
        self.retriever = retriever or HybridRetriever(
            vector_backend=build_vector_backend(self.settings),
            graph_backend=build_graph_backend(self.settings),
            top_k=self.settings.retrieval_top_k,
        )
        self.validator = validator or AnswerValidator(llm_client=llm_client)
        self.corrector = corrector or CorrectionEngine(llm_client=llm_client)
        self.generator = generator or AnswerGenerator(llm_client=llm_client)

    def run(self, query: str) -> PipelineResult:
        root = self.decomposer.decompose(query)
        state = WorkflowState(query=query, root=root, nodes=self._flatten_leaves(root))
        self._resolve_tree(state.root)
        state.nodes = self._flatten_leaves(state.root)
        state.final_answer = state.root.answer or self.generator.generate_final(state.query, state.nodes)
        return PipelineResult(
            query=state.query,
            root=state.root,
            nodes=state.nodes,
            final_answer=state.final_answer,
        )

    def _resolve_tree(self, node: QueryNode, prior_hops: list[tuple[str, str]] | None = None) -> None:
        if prior_hops is None:
            prior_hops = []
        if node.children:
            accumulated: list[tuple[str, str]] = list(prior_hops)
            for child in node.children:
                self._resolve_tree(child, accumulated)
                if child.answer:
                    accumulated.append((child.question, self._strip_sources(child.answer)))
            node.answer = self.generator.generate_final(node.question, node.children)
            node.attempts = max((child.attempts for child in node.children), default=0)
            node.status = (
                "verified"
                if all(child.status == "verified" for child in node.children)
                else "needs_review"
            )
            confidences = [
                child.validation.confidence
                for child in node.children
                if child.validation is not None
            ]
            if confidences:
                node.validation = self._build_group_validation(node.status, confidences)
            return
        self._resolve_leaf(node, prior_hops)

    def _resolve_leaf(self, node: QueryNode, prior_hops: list[tuple[str, str]] | None = None) -> None:
        prior_hops = prior_hops or []
        # Enrich the retrieval query with concrete answers from previous hops.
        # e.g. sub-q2 = "When is the composer's birthday?" + " K. V. Mahadevan" (from hop 1)
        base_question = node.question
        retrieval_question = self._enrich_query(base_question, prior_hops)
        for attempt in range(self.settings.max_retries + 1):
            node.attempts = attempt + 1
            node.documents = self.retriever.retrieve(retrieval_question)
            node.answer = self.generator.generate_for_node(
                base_question, node.documents, prior_hops=prior_hops
            )
            node.validation = self.validator.validate(node.answer, node.documents)
            if node.validation.passed:
                node.status = "verified"
                return
            retrieval_question = self.corrector.refine(retrieval_question, node.attempts)
        node.status = "needs_review"

    @staticmethod
    def _enrich_query(question: str, prior_hops: list[tuple[str, str]]) -> str:
        """Append short prior-hop answers to the retrieval query so that rare
        named entities found in hop N-1 boost retrieval scores in hop N."""
        if not prior_hops:
            return question
        extra = " ".join(ans for _, ans in prior_hops)
        return f"{question} {extra}"

    @staticmethod
    def _strip_sources(text: str) -> str:
        import re
        return re.sub(r"\s*Sources:\s.*$", "", text, flags=re.IGNORECASE).strip()

    def _flatten_leaves(self, root: QueryNode) -> list[QueryNode]:
        if root.is_leaf:
            return [root]
        leaves: list[QueryNode] = []
        for child in root.children:
            leaves.extend(self._flatten_leaves(child))
        return leaves

    def _build_group_validation(
        self, status: str, confidences: list[float]
    ) -> ValidationResult:
        average_confidence = sum(confidences) / len(confidences)
        rationale = (
            "All child nodes were verified."
            if status == "verified"
            else "At least one child node requires review."
        )
        return ValidationResult(
            passed=status == "verified",
            confidence=average_confidence,
            rationale=rationale,
        )
