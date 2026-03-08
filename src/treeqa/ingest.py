from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from treeqa.config import TreeQASettings
from treeqa.retrieval.scoring import normalize_text


@dataclass(slots=True)
class IngestReport:
    vector_chunks: int
    graph_facts: int
    vector_index_path: str
    graph_index_path: str


@dataclass(slots=True)
class IndexedChunk:
    source_id: str
    source_path: str
    content: str


@dataclass(slots=True)
class IndexedFact:
    source_id: str
    source_path: str
    content: str


def build_local_indices(settings: TreeQASettings | None = None) -> IngestReport:
    settings = settings or TreeQASettings.from_env()
    data_dir = settings.resolved_data_dir
    documents_dir = data_dir / "documents"
    graph_dir = data_dir / "graph"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    vector_index_path = (
        settings.resolve_path(settings.local_vector_index_path)
        if settings.local_vector_index_path
        else index_dir / "vector_index.jsonl"
    )
    graph_index_path = (
        settings.resolve_path(settings.local_graph_index_path)
        if settings.local_graph_index_path
        else index_dir / "graph_facts.jsonl"
    )

    chunks = _build_document_chunks(documents_dir)
    facts = _build_graph_facts(graph_dir)

    _write_jsonl(vector_index_path, [asdict(chunk) for chunk in chunks])
    _write_jsonl(graph_index_path, [asdict(fact) for fact in facts])

    return IngestReport(
        vector_chunks=len(chunks),
        graph_facts=len(facts),
        vector_index_path=str(vector_index_path),
        graph_index_path=str(graph_index_path),
    )


def _build_document_chunks(documents_dir: Path) -> list[IndexedChunk]:
    if not documents_dir.exists():
        return []
    chunks: list[IndexedChunk] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        text = _read_document(path)
        for index, chunk in enumerate(_chunk_text(text), start=1):
            chunks.append(
                IndexedChunk(
                    source_id=f"{path.stem}-chunk-{index}",
                    source_path=str(path),
                    content=chunk,
                )
            )
    return chunks


def _build_graph_facts(graph_dir: Path) -> list[IndexedFact]:
    if not graph_dir.exists():
        return []
    facts: list[IndexedFact] = []
    for path in sorted(graph_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".jsonl":
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                content = str(payload.get("content") or payload.get("fact") or "")
                if not content:
                    continue
                source_id = str(payload.get("source_id") or f"{path.stem}-fact-{index}")
                facts.append(IndexedFact(source_id=source_id, source_path=str(path), content=content))
        elif path.suffix.lower() in {".md", ".txt"}:
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                content = line.strip("- ").strip()
                if not content:
                    continue
                facts.append(
                    IndexedFact(
                        source_id=f"{path.stem}-fact-{index}",
                        source_path=str(path),
                        content=content,
                    )
                )
    return facts


def _read_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return normalize_text(text)
    payload = json.loads(text)
    if isinstance(payload, dict):
        text = "\n".join(
            str(value) for value in payload.values() if isinstance(value, (str, int, float))
        )
    elif isinstance(payload, list):
        text = "\n".join(str(item) for item in payload)
    else:
        text = str(payload)
    return normalize_text(text)


def _chunk_text(text: str, max_chars: int = 220) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    return chunks


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
