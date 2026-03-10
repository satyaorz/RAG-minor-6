from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

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
    title: str
    section: str
    chunk_index: int
    ingested_at: str
    content: str


@dataclass(slots=True)
class IndexedFact:
    source_id: str
    source_path: str
    ingested_at: str
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

    faiss_path = vector_index_path.with_suffix(".faiss")
    meta_path = vector_index_path.with_name("vector_meta.json")
    _build_faiss_index(chunks, settings.embedding_model, faiss_path, meta_path)

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
    ingested_at = _iso_now()
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        title = _extract_title(path)
        sections = _split_into_sections(path)
        chunk_index = 0
        for section_name, section_text in sections:
            for chunk in _chunk_text(section_text):
                chunk_index += 1
                chunks.append(
                    IndexedChunk(
                        source_id=f"{path.stem}-chunk-{chunk_index}",
                        source_path=str(path),
                        title=title,
                        section=section_name,
                        chunk_index=chunk_index,
                        ingested_at=ingested_at,
                        content=chunk,
                    )
                )
    return chunks


def _build_graph_facts(graph_dir: Path) -> list[IndexedFact]:
    if not graph_dir.exists():
        return []
    facts: list[IndexedFact] = []
    ingested_at = _iso_now()
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
                facts.append(IndexedFact(source_id=source_id, source_path=str(path), ingested_at=ingested_at, content=content))
        elif path.suffix.lower() in {".md", ".txt"}:
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                content = line.strip("- ").strip()
                if not content:
                    continue
                facts.append(
                    IndexedFact(
                        source_id=f"{path.stem}-fact-{index}",
                        source_path=str(path),
                        ingested_at=ingested_at,
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


def _extract_title(path: Path) -> str:
    """Return the first H1 heading found in a markdown file, else the stem."""
    if path.suffix.lower() == ".md":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    return path.stem.replace("_", " ").title()


def _split_into_sections(path: Path) -> list[tuple[str, str]]:
    """Split a document into (section_heading, section_text) pairs.

    Non-markdown files are treated as a single unnamed section.
    """
    if path.suffix.lower() not in {".md", ".txt"}:
        return [("", _read_document(path))]

    raw = path.read_text(encoding="utf-8")
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(2).strip()) for m in heading_pattern.finditer(raw)]

    if not positions:
        return [("", normalize_text(raw))]

    sections: list[tuple[str, str]] = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
        body = raw[start:end]
        # Remove the heading line itself from the body text
        body = re.sub(r"^#{1,3}\s+.+\n?", "", body, count=1)
        normalized = normalize_text(body)
        if normalized:
            sections.append((heading, normalized))
    return sections or [("", normalize_text(raw))]


def _chunk_text(text: str, max_chars: int = 300, overlap_sentences: int = 1) -> list[str]:
    """Split text into sentence-boundary chunks with optional sentence overlap."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        candidate_len = current_len + (1 if current_len else 0) + len(sentence)
        if candidate_len <= max_chars or not current_sentences:
            current_sentences.append(sentence)
            current_len = candidate_len
        else:
            chunks.append(" ".join(current_sentences))
            # Keep last `overlap_sentences` sentences for context continuity
            current_sentences = current_sentences[-overlap_sentences:] + [sentence]
            current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_faiss_index(
    chunks: list[IndexedChunk],
    model_name: str,
    faiss_path: Path,
    meta_path: Path,
) -> None:
    """Encode all chunks once and persist a FAISS IndexFlatIP + JSON metadata.

    Skipped gracefully if faiss or sentence-transformers is not installed.
    On subsequent server starts the embeddings are loaded from disk — no
    re-encoding required, making startup essentially instant.
    """
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return

    from treeqa.retrieval.scoring import normalize_text

    print(f"[ingest] Building FAISS index for {len(chunks)} chunks …")
    model = SentenceTransformer(model_name)
    texts = [
        " ".join(filter(None, [c.title, c.section, normalize_text(c.content)]))
        for c in chunks
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product = cosine for L2-normalised vectors
    index.add(embeddings)

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))

    # Metadata saved as JSON (portable, human-readable, no pickle risks)
    meta_path.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"[ingest] FAISS index saved → {faiss_path} ({index.ntotal} vectors, d={d})")
