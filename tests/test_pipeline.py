import unittest

from treeqa.pipeline import TreeQAPipeline


class TreeQAPipelineTest(unittest.TestCase):
    def test_pipeline_returns_final_answer_and_tree(self) -> None:
        pipeline = TreeQAPipeline()

        result = pipeline.run(
            "How does TreeQA use hybrid retrieval and validation for multi-hop QA?"
        )

        self.assertTrue(result.final_answer)
        self.assertEqual(result.root.node_id, "root")
        self.assertEqual(result.root.status, "verified")
        self.assertTrue(result.nodes)
        self.assertTrue(
            all(node.status in {"verified", "needs_review"} for node in result.nodes)
        )

    def test_pipeline_decomposes_multi_part_query_under_root(self) -> None:
        pipeline = TreeQAPipeline()

        result = pipeline.run(
            "What is HotpotQA and how does LangGraph support agent workflows?"
        )

        self.assertEqual(result.root.node_id, "root")
        self.assertEqual(len(result.root.children), 2)
        self.assertEqual(len(result.nodes), 2)

    def test_pipeline_uses_root_as_leaf_for_single_part_query(self) -> None:
        pipeline = TreeQAPipeline()

        result = pipeline.run("What is HotpotQA?")

        self.assertFalse(result.root.children)
        self.assertEqual(result.nodes[0].node_id, "root")


if __name__ == "__main__":
    unittest.main()
