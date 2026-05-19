from __future__ import annotations

from hamhrag.pipeline import HamhRagPipeline


def build_workflow() -> HamhRagPipeline:
    """Factory kept separate so a future orchestrator can replace it later."""

    return HamhRagPipeline()

