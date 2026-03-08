import unittest

from treeqa.agents.corrector import CorrectionEngine
from treeqa.agents.decomposer import QueryDecomposer
from treeqa.backends.llm import OpenAICompatibleLLMClient, build_llm_client
from treeqa.config import TreeQASettings
from treeqa.models import RetrievedDocument
from treeqa.retrieval.hybrid import HybridRetriever


class FakeLLMClient:
    def __init__(self, text_response: str = "", json_response: object | None = None) -> None:
        self.text_response = text_response
        self.json_response = json_response or {}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.text_response

    def generate_json(self, system_prompt: str, user_prompt: str) -> object:
        return self.json_response


class FakeVectorBackend:
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                source_id="vector-1",
                source_type="vector",
                content="Vector evidence",
                score=0.4,
            )
        ]


class FakeGraphBackend:
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                source_id="graph-1",
                source_type="graph",
                content="Graph evidence",
                score=0.8,
            )
        ]


class AdapterTest(unittest.TestCase):
    def test_build_llm_client_for_openai_provider(self) -> None:
        settings = TreeQASettings(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            openai_api_key="test-key",
        )

        client = build_llm_client(settings)

        self.assertIsInstance(client, OpenAICompatibleLLMClient)

    def test_decomposer_uses_llm_subquestions_when_available(self) -> None:
        decomposer = QueryDecomposer(
            llm_client=FakeLLMClient(
                json_response={"sub_questions": ["Who built TreeQA?", "How is it validated?"]}
            )
        )

        nodes = decomposer.decompose("Who built TreeQA and how is it validated?")

        self.assertEqual(
            [node.question for node in nodes],
            ["Who built TreeQA?", "How is it validated?"],
        )

    def test_correction_engine_uses_llm_rewrite_when_available(self) -> None:
        corrector = CorrectionEngine(
            llm_client=FakeLLMClient(text_response="TreeQA hallucination mitigation workflow")
        )

        rewritten = corrector.refine("How does TreeQA help?", 1)

        self.assertEqual(rewritten, "TreeQA hallucination mitigation workflow")

    def test_hybrid_retriever_combines_backends(self) -> None:
        retriever = HybridRetriever(
            vector_backend=FakeVectorBackend(),
            graph_backend=FakeGraphBackend(),
            top_k=2,
        )

        documents = retriever.retrieve("TreeQA evidence")

        self.assertEqual([document.source_type for document in documents], ["graph", "vector"])

    def test_build_llm_client_for_openrouter_provider(self) -> None:
        settings = TreeQASettings(
            llm_provider="openrouter",
            llm_model="meta-llama/llama-3.3-70b-instruct:free",
            llm_base_url="https://openrouter.ai/api/v1",
            openrouter_api_key="router-key",
            openrouter_site_url="http://localhost",
            openrouter_app_name="TreeQA",
        )

        client = build_llm_client(settings)

        self.assertIsInstance(client, OpenAICompatibleLLMClient)
        self.assertEqual(client.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(client.extra_headers, {"HTTP-Referer": "http://localhost", "X-Title": "TreeQA"})


if __name__ == "__main__":
    unittest.main()
