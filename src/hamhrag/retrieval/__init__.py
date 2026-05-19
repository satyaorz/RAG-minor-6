__all__ = ["HybridRetriever", "QueryRouter", "RetrievalTrace", "RoutePlan"]


def __getattr__(name: str):
    if name in {"HybridRetriever", "QueryRouter", "RetrievalTrace", "RoutePlan"}:
        from hamhrag.retrieval.hybrid import (
            HybridRetriever,
            QueryRouter,
            RetrievalTrace,
            RoutePlan,
        )

        return {
            "HybridRetriever": HybridRetriever,
            "QueryRouter": QueryRouter,
            "RetrievalTrace": RetrievalTrace,
            "RoutePlan": RoutePlan,
        }[name]
    raise AttributeError(name)
