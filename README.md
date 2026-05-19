# HAMH-RAG

HAMH-RAG (Hallucination-Aware Multi-Hop RAG) is a hallucination-aware, multi-hop RAG system with a logic-tree workflow and interactive web UI. The code package is still named `hamhrag` for now.

## Layout

- `src/hamhrag/`: core package
- `src/hamhrag/agents/`: decomposition, validation, correction, and answer generation
- `src/hamhrag/retrieval/`: hybrid retrieval interfaces
- `src/hamhrag/api/`: FastAPI backend (REST API + SPA server)
- `src/hamhrag/ui/`: Streamlit UI and the vanilla-JS SPA (`index.html`)
- `tests/`: offline unit and integration tests

## Quick start

1. Create and activate a virtual environment.
2. Install the project: `pip install -e ".[api]"`
3. Copy `.env.example` to `.env` and fill in your values.
4. Build local indices: `python -m hamhrag.cli ingest`
5. Check setup: `python -m hamhrag.cli doctor`
6. Run tests: `python -m pytest tests/ -q`

For this repo, prefer the checked command targets so every workflow uses the
same interpreter under `.venv`:

```bash
make install-full
make doctor
make test
make bench
make app
```

`make doctor`, `make test`, and `make bench` force Hugging Face/Transformers
offline mode. That prevents repeated network checks during normal development
and benchmarking. Run `make ingest` without offline mode once when you actually
want to download/cache the embedding model and rebuild the FAISS index.

## Web UI

Start the FastAPI server (serves the interactive SPA at `http://localhost:8000`):

```bash
.venv/bin/python -m pip install -e ".[api]"
.venv/bin/python -m uvicorn hamhrag.api.app:app --reload --port 8000
```

Then open `http://localhost:8000` in a browser. Type a question and press **Run ▶** (or Ctrl+Enter).
The logic tree renders each sub-question node with its status, confidence, answer, rationale, and evidence.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the SPA (index.html) |
| `POST` | `/api/run` | `{"query": "..."}` → pipeline JSON result |
| `GET` | `/api/health` | `{"status": "ok"}` |
| `GET` | `/api/docs` | Swagger UI |

## Streamlit UI (alternative)

```bash
.venv/bin/python -m streamlit run src/hamhrag/ui/streamlit_app.py
```

## CLI

```bash
.venv/bin/python -m hamhrag.cli run        # run the pipeline interactively
.venv/bin/python -m hamhrag.cli ingest     # build local retrieval indices from data/
.venv/bin/python -m hamhrag.cli doctor     # check config
.venv/bin/python -m hamhrag.cli doctor --live-llm   # verify live LLM round-trip
.venv/bin/python -m hamhrag.cli bench --dataset data/benchmark/sample.jsonl --limit 50 --mode both
```

## Benchmarking

See [BENCHMARK.md](BENCHMARK.md) for the current protocol and targets.

## Provider setup

The app now auto-loads a repo-local `.env` file at startup. Keep secrets in:

- [\.env](d:\CompD\sem6Min\RAG-minor-6\.env)

Do not commit that file. It is already ignored by [\.gitignore](d:\CompD\sem6Min\RAG-minor-6\.gitignore).

Set these env vars to enable real services:

- `HAMHRAG_LLM_PROVIDER=openai`
- `HAMHRAG_LLM_MODEL=<model name>`
- `OPENAI_API_KEY=<api key>`
- `HAMHRAG_LLM_PROVIDER=openrouter`
- `HAMHRAG_LLM_BASE_URL=https://openrouter.ai/api/v1`
- `HAMHRAG_LLM_MODEL=arcee-ai/trinity-mini:free`
- `OPENROUTER_API_KEY=<api key>`
- `HAMHRAG_VECTOR_PROVIDER=qdrant`
- `VECTOR_STORE_URL=<qdrant url>`
- `HAMHRAG_VECTOR_COLLECTION=<collection>`
- `HAMHRAG_GRAPH_PROVIDER=neo4j`
- `GRAPH_STORE_URL=<neo4j uri>`

Optional provider packages:

- `pip install -e .[providers]`

FAISS is auto-selected for the `local` vector provider when a `.faiss` index is present.
Install `faiss-cpu` (or the providers extra) and run `python -m hamhrag.cli ingest` to build the index.

### OpenRouter example

Use this configuration to target OpenRouter with the current recommended free model stack:

```env
HAMHRAG_LLM_PROVIDER=openrouter
HAMHRAG_LLM_BASE_URL=https://openrouter.ai/api/v1
HAMHRAG_LLM_MODEL=arcee-ai/trinity-mini:free
HAMHRAG_LLM_FALLBACK_MODELS=nvidia/nemotron-3-nano-30b-a3b:free,arcee-ai/trinity-large-preview:free,qwen/qwen3-4b:free,openrouter/free
OPENROUTER_API_KEY=your-key
HAMHRAG_OPENROUTER_SITE_URL=http://localhost
HAMHRAG_OPENROUTER_APP_NAME=HAMH-RAG
```

### Minimal real setup for your current case

Since you only have an OpenRouter key right now, start with OpenRouter plus local file-backed retrieval. That is enough to run the project with a real LLM while avoiding external database setup.

Use this in your local [\.env](d:\CompD\sem6Min\RAG-minor-6\.env):

```env
HAMHRAG_LLM_PROVIDER=openrouter
HAMHRAG_LLM_BASE_URL=https://openrouter.ai/api/v1
HAMHRAG_LLM_MODEL=arcee-ai/trinity-mini:free
HAMHRAG_LLM_FALLBACK_MODELS=nvidia/nemotron-3-nano-30b-a3b:free,arcee-ai/trinity-large-preview:free,qwen/qwen3-4b:free,openrouter/free
OPENROUTER_API_KEY=your_openrouter_key
HAMHRAG_OPENROUTER_SITE_URL=http://localhost
HAMHRAG_OPENROUTER_APP_NAME=HAMH-RAG

HAMHRAG_VECTOR_PROVIDER=local
HAMHRAG_GRAPH_PROVIDER=local

HAMHRAG_MAX_RETRIES=2
HAMHRAG_RETRIEVAL_TOP_K=3
HAMHRAG_LLM_TIMEOUT_SECONDS=30
HAMHRAG_LLM_TEMPERATURE=0.0
HAMHRAG_DATA_DIR=data
```

You can leave the rest blank for now.

Put your source files under:

- [data/documents](d:\CompD\sem6Min\RAG-minor-6\data\documents)
- [data/graph](d:\CompD\sem6Min\RAG-minor-6\data\graph)

Then build the local indices:

```bash
python -m hamhrag.cli ingest
```

Before building further, run:

```bash
python -m hamhrag.cli doctor
python -m hamhrag.cli doctor --live-llm
```

`doctor` validates configuration without network calls. `doctor --live-llm` performs a real LLM probe and will surface issues such as invalid keys, rate limits, or blocked network access.

When multiple fallback models are configured, HAMH-RAG will try them in order until one succeeds.

### When you need more than that

- Add Qdrant when you want real document retrieval over a corpus.
- Add Neo4j when you want structured fact retrieval and graph traversal.
- Use `local` providers first so you can iterate on prompts and workflow behavior with real project documents.
