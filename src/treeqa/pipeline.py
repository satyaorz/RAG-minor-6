from __future__ import annotations

from treeqa.agents import (
    AnswerGenerator,
    AnswerValidator,
    CorrectionEngine,
    QueryDecomposer,
)
from treeqa.backends import build_graph_backend, build_llm_client, build_vector_backend
from treeqa.config import TreeQASettings
from treeqa.models import PipelineResult, QueryNode
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
        state = WorkflowState(query=query, nodes=self.decomposer.decompose(query))
        for node in state.nodes:
            self._resolve_node(node)
        state.final_answer = self.generator.generate_final(state.query, state.nodes)
        return PipelineResult(
            query=state.query,
            nodes=state.nodes,
            final_answer=state.final_answer,
        )

    def _resolve_node(self, node: QueryNode) -> None:
        question = node.question
        for attempt in range(self.settings.max_retries + 1):
            node.attempts = attempt + 1
            node.documents = self.retriever.retrieve(question)
            node.answer = self.generator.generate_for_node(question, node.documents)
            node.validation = self.validator.validate(node.answer, node.documents)
            if node.validation.passed:
                node.status = "verified"
                return
            question = self.corrector.refine(question, node.attempts)
        node.status = "needs_review"
