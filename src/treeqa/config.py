from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class TreeQASettings:
    max_retries: int = 2
    retrieval_top_k: int = 3
    data_dir: str = "data"
    llm_provider: str = "stub"
    llm_model: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: int = 30
    llm_temperature: float = 0.0
    openrouter_api_key: str = ""
    openrouter_site_url: str = ""
    openrouter_app_name: str = "TreeQA"
    vector_store_url: str = ""
    vector_provider: str = "memory"
    vector_collection: str = ""
    vector_api_key: str = ""
    graph_store_url: str = ""
    graph_provider: str = "memory"
    graph_database: str = ""
    graph_username: str = ""
    graph_password: str = ""
    graph_index_name: str = "entity_index"
    graph_query: str = (
        "CALL db.index.fulltext.queryNodes($index_name, $question) "
        "YIELD node, score "
        "RETURN coalesce(node.id, node.name, node.title, toString(id(node))) AS source_id, "
        "coalesce(node.summary, node.description, node.value, '') AS content, score "
        "LIMIT $limit"
    )
    openai_api_key: str = ""

    @classmethod
    def from_env(cls) -> "TreeQASettings":
        return cls(
            max_retries=int(os.getenv("TREEQA_MAX_RETRIES", "2")),
            retrieval_top_k=int(os.getenv("TREEQA_RETRIEVAL_TOP_K", "3")),
            data_dir=os.getenv("TREEQA_DATA_DIR", "data"),
            llm_provider=os.getenv("TREEQA_LLM_PROVIDER", "stub"),
            llm_model=os.getenv("TREEQA_LLM_MODEL", ""),
            llm_base_url=os.getenv("TREEQA_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_timeout_seconds=int(os.getenv("TREEQA_LLM_TIMEOUT_SECONDS", "30")),
            llm_temperature=float(os.getenv("TREEQA_LLM_TEMPERATURE", "0.0")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_site_url=os.getenv("TREEQA_OPENROUTER_SITE_URL", ""),
            openrouter_app_name=os.getenv("TREEQA_OPENROUTER_APP_NAME", "TreeQA"),
            vector_store_url=os.getenv("VECTOR_STORE_URL", ""),
            vector_provider=os.getenv("TREEQA_VECTOR_PROVIDER", "memory"),
            vector_collection=os.getenv("TREEQA_VECTOR_COLLECTION", ""),
            vector_api_key=os.getenv("TREEQA_VECTOR_API_KEY", ""),
            graph_store_url=os.getenv("GRAPH_STORE_URL", ""),
            graph_provider=os.getenv("TREEQA_GRAPH_PROVIDER", "memory"),
            graph_database=os.getenv("TREEQA_GRAPH_DATABASE", ""),
            graph_username=os.getenv("TREEQA_GRAPH_USERNAME", ""),
            graph_password=os.getenv("TREEQA_GRAPH_PASSWORD", ""),
            graph_index_name=os.getenv("TREEQA_GRAPH_INDEX_NAME", "entity_index"),
            graph_query=os.getenv(
                "TREEQA_GRAPH_QUERY",
                (
                    "CALL db.index.fulltext.queryNodes($index_name, $question) "
                    "YIELD node, score "
                    "RETURN coalesce(node.id, node.name, node.title, toString(id(node))) AS source_id, "
                    "coalesce(node.summary, node.description, node.value, '') AS content, score "
                    "LIMIT $limit"
                ),
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
