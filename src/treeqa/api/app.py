"""
TreeQA FastAPI backend.

Start:
    uvicorn treeqa.api.app:app --reload --port 8000

Endpoints:
    GET  /              → SPA (index.html)
    GET  /api/health    → {"status": "ok"}
    GET  /api/documents → list files in data/documents/
    POST /api/upload    → upload + ingest a .txt/.md file
    POST /api/run       → run TreeQA pipeline, return PipelineResult
    POST /api/compare   → run traditional RAG + TreeQA in parallel
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import pathlib

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from treeqa.config import TreeQASettings
from treeqa.ingest import build_local_indices
from treeqa.pipeline import TreeQAPipeline

_PIPELINE_TIMEOUT = 110.0  # seconds — slightly under the 2-min client timeout
_executor = ThreadPoolExecutor(max_workers=4)

_UI_HTML = pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html"
_settings = TreeQASettings.from_env()
_DOCS_DIR = _settings.resolved_data_dir / "documents"
_ALLOWED_EXTENSIONS = {".txt", ".md"}

app = FastAPI(title="TreeQA", version="0.1.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline — built once at startup so index files are loaded once.
_pipeline: TreeQAPipeline | None = None


@app.on_event("startup")
def _startup() -> None:
    global _pipeline, _settings
    _settings = TreeQASettings.from_env()
    _pipeline = TreeQAPipeline(settings=_settings)


def _reload_pipeline() -> None:
    """Re-build the singleton pipeline (called after ingestion of new docs)."""
    global _pipeline, _settings
    _settings = TreeQASettings.from_env()
    _pipeline = TreeQAPipeline(settings=_settings)


class RunRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


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
    await loop.run_in_executor(_executor, build_local_indices, _settings)
    _reload_pipeline()
    return JSONResponse(content={
        "ok": True,
        "filename": safe_name,
        "size": len(content),
        "message": f"'{safe_name}' ingested and index updated.",
    })


@app.post("/api/run")
async def run_query(body: RunRequest) -> JSONResponse:
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _pipeline.run, body.query),
            timeout=_PIPELINE_TIMEOUT,
        )
        return JSONResponse(content=asdict(result))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {int(_PIPELINE_TIMEOUT)}s. Try a simpler query.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_traditional_sync(query: str) -> dict:
    """Traditional RAG baseline: retrieve only, no LLM generation or validation.

    Represents the simplest RAG approach — keyword search returns top documents
    and concatenates their content as the "answer".  No multi-hop decomposition,
    no hallucination validation, no confidence scoring from an LLM.
    """
    docs = _pipeline.retriever.retrieve(query)
    # Directly use retrieved content — no LLM call, representing the naive baseline
    if docs:
        top_score = round(docs[0].score, 4)
        answer = " … ".join(d.content[:300].strip() for d in docs[:3])
    else:
        top_score = 0.0
        answer = "No relevant documents found."
    return {
        "query": query,
        "answer": answer,
        "confidence": top_score,          # retrieval score as proxy confidence
        "passed": top_score >= 0.5,
        "rationale": f"Returned top {len(docs)} document(s) by retrieval score. No LLM generation or hallucination check.",
        "doc_count": len(docs),
        "documents": [asdict(d) for d in docs],
    }


@app.post("/api/compare")
async def compare_query(body: RunRequest) -> JSONResponse:
    """Run traditional RAG (retrieval-only, instant) then TreeQA; return side-by-side."""
    loop = asyncio.get_event_loop()
    try:
        # Traditional RAG: pure retrieval, no LLM — runs in <1s on a thread
        trad = await loop.run_in_executor(_executor, _run_traditional_sync, body.query)
        # TreeQA: full multi-hop pipeline with LLM calls — may take up to PIPELINE_TIMEOUT
        tqa = await asyncio.wait_for(
            loop.run_in_executor(_executor, _pipeline.run, body.query),
            timeout=_PIPELINE_TIMEOUT,
        )
        return JSONResponse(content={"traditional": trad, "treeqa": asdict(tqa)})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="TreeQA pipeline timed out. Try a simpler query.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
