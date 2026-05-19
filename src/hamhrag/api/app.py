"""
HAMH-RAG FastAPI backend (Hallucination-Aware Multi-Hop RAG).

Start:
    uvicorn hamhrag.api.app:app --reload --port 8000

Endpoints:
    GET  /                    → SPA (index.html)
    GET  /api/health          → {"status": "ok"}
    GET  /api/documents       → list files in data/documents/
    GET  /api/datasets        → list supported benchmark datasets
    POST /api/upload          → upload + ingest a .txt/.md file
    POST /api/run             → run HAMH-RAG pipeline, return PipelineResult
    POST /api/run/stream      → same, streamed as Server-Sent Events
    POST /api/run_rag         → run Normal RAG (single-hop), return result
    POST /api/run_rag/stream  → same, streamed as Server-Sent Events
    POST /api/compare         → run traditional RAG + HAMH-RAG in parallel
    POST /api/load_dataset    → stream a HF benchmark dataset and re-index
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import logging
import os
import pathlib
from functools import partial

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from hamhrag.config import HamhRagSettings
from hamhrag.dataset_loader import SUPPORTED_DATASETS, load_and_write_corpus
from hamhrag.ingest import build_local_indices
from hamhrag.pipeline import HamhRagPipeline

_PIPELINE_TIMEOUT = float(os.getenv("HAMHRAG_PIPELINE_TIMEOUT", "110"))  # seconds
_executor = ThreadPoolExecutor(max_workers=4)

_UI_HTML = pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html"
_settings = HamhRagSettings.from_env()
_DOCS_DIR = _settings.resolved_data_dir / "documents"
_ALLOWED_EXTENSIONS = {".txt", ".md"}

app = FastAPI(title="HAMH-RAG", version="0.1.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline — built once at startup so index files are loaded once.
_pipeline: HamhRagPipeline | None = None
_log = logging.getLogger("hamhrag.api")


def _build_pipeline() -> HamhRagPipeline:
    """Blocking pipeline construction (loads embeddings, index files, etc.)."""
    global _settings
    _settings = HamhRagSettings.from_env()
    _log.info("Loading vector index and embedding model — this may take a minute...")
    pipeline = HamhRagPipeline(settings=_settings)
    _log.info("Pipeline ready.")
    return pipeline


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline
    loop = asyncio.get_event_loop()
    _pipeline = await loop.run_in_executor(_executor, _build_pipeline)


def _reload_pipeline() -> None:
    """Re-build the singleton pipeline (called after ingestion of new docs)."""
    global _pipeline
    _pipeline = _build_pipeline()


def _ensure_pipeline() -> HamhRagPipeline:
    """Ensure the singleton pipeline is ready (lazy-load if startup is still warming)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def _resolve_timeout_seconds(requested: int | None) -> float:
    if requested is None:
        return _PIPELINE_TIMEOUT
    return float(max(15, min(requested, 900)))


def _pipeline_budget(timeout_seconds: float) -> float:
    # Reserve a tiny envelope for transport/serialization around pipeline execution.
    return max(5.0, timeout_seconds - 2.0)


class RunRequest(BaseModel):
    query: str
    timeout_seconds: int | None = None
    max_retries: int | None = None
    tree_retries: int | None = None
    max_restructures: int | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_seconds_must_be_reasonable(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return max(15, min(value, 900))

    @field_validator("max_retries", "tree_retries", "max_restructures")
    @classmethod
    def retry_fields_must_be_reasonable(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return max(0, min(value, 5))


class LoadDatasetRequest(BaseModel):
    dataset_key: str
    split: str = "train"
    max_rows: int = 300

    @field_validator("dataset_key")
    @classmethod
    def key_must_be_known(cls, v: str) -> str:
        if v not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset key '{v}'")
        return v

    @field_validator("split")
    @classmethod
    def split_must_be_valid(cls, v: str) -> str:
        if v not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{v}'")
        return v

    @field_validator("max_rows")
    @classmethod
    def rows_must_be_positive(cls, v: int) -> int:
        return max(10, min(v, 2000))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_spa() -> HTMLResponse:
    if not _UI_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=_UI_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> JSONResponse:
    """Return metadata for all uploadable documents in data/documents/."""
    docs: list[dict] = []
    if _DOCS_DIR.exists():
        for f in sorted(_DOCS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in _ALLOWED_EXTENSIONS:
                docs.append({"name": f.name, "size": f.stat().st_size, "suffix": f.suffix.lower()})
    return JSONResponse(content={"documents": docs})


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a .txt or .md file, ingest it, and reload the pipeline."""
    suffix = pathlib.Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{suffix}'. Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )
    # Sanitize: only the bare filename, no path traversal
    safe_name = pathlib.Path(file.filename).name
    dest = _DOCS_DIR / safe_name
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest.write_bytes(content)

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(_executor, build_local_indices, _settings)
    _reload_pipeline()
    return JSONResponse(content={
        "ok": True,
        "filename": safe_name,
        "size": len(content),
        "vector_chunks": report.vector_chunks,
        "graph_facts": report.graph_facts,
        "message": (
            f"'{safe_name}' ingested — "
            f"{report.vector_chunks} vector chunks (unstructured) + "
            f"{report.graph_facts} graph facts (structured) indexed."
        ),
    })


@app.post("/api/upload_batch")
async def upload_documents(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Upload multiple .txt/.md files, ingest once, and reload the pipeline."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    bad = []
    for f in files:
        suffix = pathlib.Path(f.filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            bad.append(f.filename)
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported types: {', '.join(bad)}. Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    total_size = 0
    for f in files:
        safe_name = pathlib.Path(f.filename).name
        dest = _DOCS_DIR / safe_name
        content = await f.read()
        dest.write_bytes(content)
        saved.append(safe_name)
        total_size += len(content)

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(_executor, build_local_indices, _settings)
    _reload_pipeline()
    return JSONResponse(content={
        "ok": True,
        "saved_files": saved,
        "total_size": total_size,
        "vector_chunks": report.vector_chunks,
        "graph_facts": report.graph_facts,
        "message": (
            f"{len(saved)} file(s) ingested — "
            f"{report.vector_chunks} vector chunks (unstructured) + "
            f"{report.graph_facts} graph facts (structured) indexed."
        ),
    })


@app.post("/api/run")
async def run_query(body: RunRequest) -> JSONResponse:
    pipeline = _ensure_pipeline()
    loop = asyncio.get_event_loop()
    request_timeout = _resolve_timeout_seconds(body.timeout_seconds)
    pipeline_timeout = _pipeline_budget(request_timeout)
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                partial(
                    pipeline.run,
                    body.query,
                    max_seconds=pipeline_timeout,
                    max_retries_override=body.max_retries,
                    tree_retries_override=body.tree_retries,
                    max_restructures_override=body.max_restructures,
                ),
            ),
            timeout=request_timeout,
        )
        return JSONResponse(content=asdict(result))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {int(request_timeout)}s. Try a simpler query.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/run/stream")
async def run_query_stream(body: RunRequest) -> StreamingResponse:
    """HAMH-RAG pipeline with real-time Server-Sent Events progress stream.

    Events delivered as ``data: <JSON>\\n\\n`` lines:
        start        — pipeline started
        tree_attempt — tree-level attempt N / total
        decomposed   — sub-questions list
        node_start   — a leaf node is being resolved
        node_route   — retrieval route chosen for a node
        node_done    — leaf resolved (status, answer, attempts)
        node_restructure — a leaf node is being restructured
        tree_retry   — whole-tree retry after needs_review
        result       — full PipelineResult JSON (terminal event)
        error        — fatal error (terminal event)
    """
    import json as _json
    import queue as _q

    request_timeout = _resolve_timeout_seconds(body.timeout_seconds)
    pipeline_timeout = _pipeline_budget(request_timeout)

    q: _q.Queue = _q.Queue()
    _SENTINEL = object()

    def _callback(**kwargs):
        q.put(kwargs)

    def _run():
        try:
            pipeline = _ensure_pipeline()
            result = pipeline.run(
                body.query,
                progress_callback=_callback,
                max_seconds=pipeline_timeout,
                max_retries_override=body.max_retries,
                tree_retries_override=body.tree_retries,
                max_restructures_override=body.max_restructures,
            )
            q.put({"event": "result", "data": asdict(result)})
        except Exception as exc:
            q.put({"event": "error", "message": str(exc)})
        finally:
            q.put(_SENTINEL)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run)

    async def _generate():
        import time as _time
        deadline = _time.monotonic() + request_timeout
        while True:
            if _time.monotonic() > deadline:
                yield f"data: {_json.dumps({'event': 'error', 'message': f'Pipeline timed out after {int(request_timeout)}s. Try a simpler query.'}, default=str)}\n\n"
                break
            try:
                item = q.get_nowait()
            except _q.Empty:
                await asyncio.sleep(0.04)
                continue
            if item is _SENTINEL:
                break
            yield f"data: {_json.dumps(item, default=str)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


def _run_traditional_sync(query: str) -> dict:
    """Normal RAG baseline: vector-only top-k retrieval + LLM generation."""
    pipeline = _ensure_pipeline()
    top_k = max(1, int(getattr(pipeline.retriever, "top_k", pipeline.settings.retrieval_top_k)))
    vector_backend = getattr(pipeline.retriever, "vector_backend", None)
    if vector_backend is None or not hasattr(vector_backend, "search"):
        docs = []
    else:
        try:
            docs = vector_backend.search(query, top_k)
        except Exception as exc:
            _log.warning("Normal RAG vector search failed: %s", exc)
            docs = []
    if docs:
        top_score = round(docs[0].score, 4)
        answer = pipeline.generator.generate_for_node(query, docs)
    else:
        top_score = 0.0
        answer = "No relevant documents found."
    return {
        "query": query,
        "answer": answer,
        "confidence": top_score,
        "passed": top_score >= 0.5,
        "rationale": f"Single-hop vector retrieval + LLM generation over top {len(docs)} document(s). No decomposition, graph lookup, or validation loop.",
        "doc_count": len(docs),
        "documents": [asdict(d) for d in docs],
    }


@app.post("/api/run_rag")
async def run_rag(body: RunRequest) -> JSONResponse:
    """Normal RAG: single-hop retrieve + LLM generate. No decomposition."""
    loop = asyncio.get_event_loop()
    request_timeout = _resolve_timeout_seconds(body.timeout_seconds)
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_traditional_sync, body.query),
            timeout=request_timeout,
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"RAG timed out after {int(request_timeout)}s.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/run_rag/stream")
async def run_rag_stream(body: RunRequest) -> StreamingResponse:
    """Normal RAG with SSE stream — emits start → result events."""
    import json as _json
    import queue as _q

    request_timeout = _resolve_timeout_seconds(body.timeout_seconds)

    q: _q.Queue = _q.Queue()
    _SENTINEL = object()

    def _run():
        try:
            q.put({"event": "start", "query": body.query})
            result = _run_traditional_sync(body.query)
            q.put({"event": "result", "data": result})
        except Exception as exc:
            q.put({"event": "error", "message": str(exc)})
        finally:
            q.put(_SENTINEL)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run)

    async def _generate():
        import time as _time
        deadline = _time.monotonic() + request_timeout
        while True:
            if _time.monotonic() > deadline:
                yield f"data: {_json.dumps({'event': 'error', 'message': f'RAG timed out after {int(request_timeout)}s.'}, default=str)}\n\n"
                break
            try:
                item = q.get_nowait()
            except _q.Empty:
                await asyncio.sleep(0.04)
                continue
            if item is _SENTINEL:
                break
            yield f"data: {_json.dumps(item, default=str)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/api/compare")
async def compare_query(body: RunRequest) -> JSONResponse:
    """Run traditional RAG and HamhRag concurrently; return side-by-side."""
    loop = asyncio.get_event_loop()
    request_timeout = _resolve_timeout_seconds(body.timeout_seconds)
    pipeline_timeout = _pipeline_budget(request_timeout)
    try:
        pipeline = _ensure_pipeline()
        trad_future = loop.run_in_executor(_executor, _run_traditional_sync, body.query)
        tqa_future = loop.run_in_executor(
            _executor,
            partial(
                pipeline.run,
                body.query,
                max_seconds=pipeline_timeout,
                max_retries_override=body.max_retries,
                tree_retries_override=body.tree_retries,
                max_restructures_override=body.max_restructures,
            ),
        )
        trad, tqa = await asyncio.wait_for(
            asyncio.gather(trad_future, tqa_future),
            timeout=request_timeout,
        )
        return JSONResponse(content={"traditional": trad, "hamhrag": asdict(tqa)})
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"HamhRag pipeline timed out after {int(request_timeout)}s. Try a simpler query.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/benchmark")
def list_benchmark_qa() -> JSONResponse:
    """Return all saved benchmark Q&A pairs from data/benchmark/*.jsonl, grouped by file."""
    bench_dir = _settings.resolved_data_dir / "benchmark"
    groups: list[dict] = []
    if bench_dir.exists():
        import json as _json
        for path in sorted(bench_dir.glob("*.jsonl")):
            pairs: list[dict] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    if isinstance(obj, dict) and obj.get("question"):
                        pairs.append({
                            "question": str(obj.get("question", "")),
                            "answer":   str(obj.get("answer", "")),
                        })
                except Exception:
                    pass
            if pairs:
                groups.append({"file": path.stem, "pairs": pairs})
    return JSONResponse(content={"groups": groups})


@app.get("/api/datasets")
def list_datasets() -> JSONResponse:
    """Return metadata for all supported benchmark datasets."""
    return JSONResponse(content={"datasets": SUPPORTED_DATASETS})


def _load_dataset_sync(dataset_key: str, split: str, max_rows: int) -> dict:
    """Blocking: download corpus from HF, write markdown files, re-index, reload pipeline."""
    result = load_and_write_corpus(
        dataset_key=dataset_key,
        split=split,
        max_rows=max_rows,
        data_dir=_settings.resolved_data_dir,
    )
    report = build_local_indices(_settings)
    _reload_pipeline()
    return {
        "ok": True,
        "dataset_key": result.dataset_key,
        "dataset_name": result.dataset_name,
        "split": result.split,
        "rows_processed": result.rows_processed,
        "docs_written": result.docs_written,
        "vector_chunks": report.vector_chunks,
        "graph_facts": report.graph_facts,
        "sample_questions": result.sample_questions,
        "message": (
            f"Loaded {result.rows_processed} {result.dataset_name} rows \u2192 "
            f"{result.docs_written} documents "
            f"({report.vector_chunks} vector chunks + {report.graph_facts} graph facts) indexed."
        ),
    }


@app.post("/api/load_dataset")
async def load_dataset_endpoint(body: LoadDatasetRequest) -> JSONResponse:
    """Stream a HuggingFace benchmark dataset, extract corpus, re-index, reload pipeline.

    Requires the ``datasets`` package (``pip install datasets``).
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _load_dataset_sync,
                body.dataset_key,
                body.split,
                body.max_rows,
            ),
            timeout=300.0,  # 5 min cap for large dataset downloads
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Dataset loading timed out (5 min). Try fewer rows.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
