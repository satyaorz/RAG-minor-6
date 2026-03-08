__all__ = ["HybridRetriever"]


def __getattr__(name: str):
    if name == "HybridRetriever":
        from treeqa.retrieval.hybrid import HybridRetriever

        return HybridRetriever
    raise AttributeError(name)
