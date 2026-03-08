import unittest

from treeqa.pipeline import TreeQAPipeline


class TreeQAPipelineTest(unittest.TestCase):
    def test_pipeline_returns_final_answer_and_nodes(self) -> None:
        pipeline = TreeQAPipeline()

        result = pipeline.run(
            "How does TreeQA use hybrid retrieval and validation for multi-hop QA?"
        )

        self.assertTrue(result.final_answer)
        self.assertTrue(result.nodes)
        self.assertTrue(
            all(node.status in {"verified", "needs_review"} for node in result.nodes)
        )

    def test_pipeline_decomposes_multi_part_query(self) -> None:
        pipeline = TreeQAPipeline()

        result = pipeline.run(
            "What is HotpotQA and how does LangGraph support agent workflows?"
        )

        self.assertEqual(len(result.nodes), 2)


if __name__ == "__main__":
    unittest.main()
