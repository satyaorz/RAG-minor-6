from __future__ import annotations

import re
from typing import Callable

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

# Callable[[str, **Any], None] — pipeline events are keyword-only
ProgressCallback = Callable[..., None]


def _noop(**_kwargs) -> None:
    pass


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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str, progress_callback: ProgressCallback | None = None) -> PipelineResult:
        """Run HAMH-RAG on *query*, optionally streaming progress events via *progress_callback*.

        Tree-level retry: if the fully resolved tree status is ``needs_review``
        and ``settings.tree_retries > 0``, the query is refined by the corrector,
        the decomposition tree is re-built, and the pipeline re-runs up to
        ``tree_retries`` additional times.

        *progress_callback* is called with keyword arguments:
            event (str)    — one of start / tree_attempt / decomposed /
                             node_start / node_done / tree_retry / done
            plus event-specific fields (question, nodes, status, answer, …)
        """
        _cb: ProgressCallback = progress_callback or _noop

        _cb(event="start", query=query)

        active_query = query
        result: PipelineResult | None = None

        for tree_attempt in range(self.settings.tree_retries + 1):
            if tree_attempt > 0:
                # Refine the root query and rebuild the decomposition tree
                active_query = self.corrector.refine(active_query, tree_attempt)
                _cb(event="tree_retry", attempt=tree_attempt, refined_query=active_query)

            _cb(event="tree_attempt", attempt=tree_attempt + 1,
                total_attempts=self.settings.tree_retries + 1)

            root = self.decomposer.decompose(active_query)
            state = WorkflowState(
                query=active_query, root=root, nodes=self._flatten_leaves(root)
            )
            _cb(event="decomposed",
                nodes=[n.question for n in state.nodes],
                total=len(state.nodes))

            self._resolve_tree(state.root, progress_callback=_cb)
            state.nodes = self._flatten_leaves(state.root)
            state.final_answer = (
                state.root.answer
                or self.generator.generate_final(state.query, state.nodes)
            )

            result = PipelineResult(
                query=state.query,
                root=state.root,
                nodes=state.nodes,
                final_answer=state.final_answer,
            )

            # Exit early if the tree was fully verified
            if state.root.status == "verified":
                break

        _cb(event="done", answer=result.final_answer, status=result.root.status)  # type: ignore[union-attr]
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tree / leaf resolution
    # ------------------------------------------------------------------

    def _resolve_tree(
        self,
        node: QueryNode,
        prior_hops: list[tuple[str, str]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        _cb: ProgressCallback = progress_callback or _noop
        if prior_hops is None:
            prior_hops = []
        if node.children:
            accumulated: list[tuple[str, str]] = list(prior_hops)
            for child in node.children:
                self._resolve_tree(child, accumulated, progress_callback=_cb)
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
        self._resolve_leaf(node, prior_hops, progress_callback=_cb)

    def _resolve_leaf(
        self,
        node: QueryNode,
        prior_hops: list[tuple[str, str]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        _cb: ProgressCallback = progress_callback or _noop
        prior_hops = prior_hops or []
        base_question = node.question
        retrieval_question = self._enrich_query(base_question, prior_hops)

        _cb(event="node_start", node_id=node.node_id, question=base_question)

        for attempt in range(self.settings.max_retries + 1):
            node.attempts = attempt + 1
            node.documents = self.retriever.retrieve(retrieval_question)
            node.answer = self.generator.generate_for_node(
                base_question, node.documents, prior_hops=prior_hops
            )
            node.validation = self.validator.validate(node.answer, node.documents)
            if node.validation.passed:
                node.status = "verified"
                break
            retrieval_question = self.corrector.refine(retrieval_question, node.attempts)

        if not node.validation or not node.validation.passed:
            node.status = "needs_review"

        _cb(event="node_done",
            node_id=node.node_id,
            question=base_question,
            status=node.status,
            answer=self._strip_sources(node.answer),
            attempts=node.attempts)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

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
