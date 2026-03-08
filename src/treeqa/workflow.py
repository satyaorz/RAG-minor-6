from __future__ import annotations

from treeqa.pipeline import TreeQAPipeline


def build_workflow() -> TreeQAPipeline:
    """Factory kept separate so a LangGraph implementation can replace it later."""

    return TreeQAPipeline()

