from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load key-value pairs from a local .env file without overriding real env vars."""

    path = dotenv_path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


load_dotenv()


@dataclass(slots=True)
class TreeQASettings:
    max_retries: int = 2
    retrieval_top_k: int = 6
    data_dir: str = "data"
    llm_provider: str = "stub"
    llm_model: str = ""
    llm_fallback_models: tuple[str, ...] = ()
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
    local_vector_index_path: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    graph_store_url: str = ""
    graph_provider: str = "memory"
    graph_database: str = ""
    graph_username: str = ""
    graph_password: str = ""
    graph_index_name: str = "entity_index"
    local_graph_index_path: str = ""
    graph_query: str = (
        "CALL db.index.fulltext.queryNodes($index_name, $question) "
        "YIELD node, score "
        "RETURN coalesce(node.id, node.name, node.title, toString(id(node))) AS source_id, "
        "coalesce(node.summary, node.description, node.value, '') AS content, score "
        "LIMIT $limit"
    )
    openai_api_key: str = ""

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    @classmethod
    def from_env(cls) -> "TreeQASettings":
        return cls(
            max_retries=int(os.getenv("TREEQA_MAX_RETRIES", "2")),
            retrieval_top_k=int(os.getenv("TREEQA_RETRIEVAL_TOP_K", "6")),
            data_dir=os.getenv("TREEQA_DATA_DIR", "data"),
            llm_provider=os.getenv("TREEQA_LLM_PROVIDER", "stub"),
            llm_model=os.getenv("TREEQA_LLM_MODEL", ""),
            llm_fallback_models=_split_csv_env("TREEQA_LLM_FALLBACK_MODELS"),
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
            local_vector_index_path=os.getenv("TREEQA_LOCAL_VECTOR_INDEX_PATH", ""),
            embedding_model=os.getenv("TREEQA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            graph_store_url=os.getenv("GRAPH_STORE_URL", ""),
            graph_provider=os.getenv("TREEQA_GRAPH_PROVIDER", "memory"),
            graph_database=os.getenv("TREEQA_GRAPH_DATABASE", ""),
            graph_username=os.getenv("TREEQA_GRAPH_USERNAME", ""),
            graph_password=os.getenv("TREEQA_GRAPH_PASSWORD", ""),
            graph_index_name=os.getenv("TREEQA_GRAPH_INDEX_NAME", "entity_index"),
            local_graph_index_path=os.getenv("TREEQA_LOCAL_GRAPH_INDEX_PATH", ""),
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


def _split_csv_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())
