# Project TreeQA

Scaffold for a hallucination-aware, multi-hop RAG system with a logic-tree workflow.

## Layout

- `src/treeqa/`: core package
- `src/treeqa/agents/`: decomposition, validation, correction, and answer generation
- `src/treeqa/retrieval/`: hybrid retrieval interfaces
- `src/treeqa/ui/`: Streamlit UI entrypoint
- `tests/`: baseline tests

## Quick start

1. Create and activate a virtual environment.
2. Install the project in editable mode: `pip install -e .`
3. Create a local `.env` by copying `.env.example` and filling in your real values.
4. Build local indices from `data/`: `python -m treeqa.cli ingest`
5. Run tests: `python -m unittest discover -s tests`
6. Inspect setup: `python -m treeqa.cli doctor`
7. Run the demo CLI: `python -m treeqa.cli run`
8. Run the UI: `streamlit run src/treeqa/ui/streamlit_app.py`

## Current status

This is a scaffold. It provides:

- typed domain models for a logic-tree workflow
- configurable LLM, vector, and graph adapters with offline fallbacks
- a hybrid retriever that can target memory, Qdrant, and Neo4j backends
- a validation and correction loop that can use either heuristics or an LLM judge
- a Streamlit demo for exploring the workflow

It now supports production wiring for:

- OpenAI-compatible chat models over HTTP
- Qdrant vector search through `qdrant-client`
- Neo4j graph retrieval through the official driver

Default behavior remains offline and in-memory unless provider env vars are set.
It also supports a local filesystem-backed retrieval path so you can work over real project files before introducing external databases.

## Validation

The included test suite uses the Python standard library so it runs without extra test tooling.

## Provider setup

The app now auto-loads a repo-local `.env` file at startup. Keep secrets in:

- [\.env](d:\CompD\sem6Min\RAG-minor-6\.env)

Do not commit that file. It is already ignored by [\.gitignore](d:\CompD\sem6Min\RAG-minor-6\.gitignore).

Set these env vars to enable real services:

- `TREEQA_LLM_PROVIDER=openai`
- `TREEQA_LLM_MODEL=<model name>`
- `OPENAI_API_KEY=<api key>`
- `TREEQA_LLM_PROVIDER=openrouter`
- `TREEQA_LLM_BASE_URL=https://openrouter.ai/api/v1`
- `TREEQA_LLM_MODEL=arcee-ai/trinity-mini:free`
- `OPENROUTER_API_KEY=<api key>`
- `TREEQA_VECTOR_PROVIDER=qdrant`
- `VECTOR_STORE_URL=<qdrant url>`
- `TREEQA_VECTOR_COLLECTION=<collection>`
- `TREEQA_GRAPH_PROVIDER=neo4j`
- `GRAPH_STORE_URL=<neo4j uri>`

Optional provider packages:

- `pip install -e .[providers]`

### OpenRouter example

Use this configuration to target OpenRouter with the current recommended free model stack:

```env
TREEQA_LLM_PROVIDER=openrouter
TREEQA_LLM_BASE_URL=https://openrouter.ai/api/v1
TREEQA_LLM_MODEL=arcee-ai/trinity-mini:free
TREEQA_LLM_FALLBACK_MODELS=nvidia/nemotron-3-nano-30b-a3b:free,arcee-ai/trinity-large-preview:free,qwen/qwen3-4b:free,openrouter/free
OPENROUTER_API_KEY=your-key
TREEQA_OPENROUTER_SITE_URL=http://localhost
TREEQA_OPENROUTER_APP_NAME=TreeQA
```

### Minimal real setup for your current case

Since you only have an OpenRouter key right now, start with OpenRouter plus local file-backed retrieval. That is enough to run the project with a real LLM while avoiding external database setup.

Use this in your local [\.env](d:\CompD\sem6Min\RAG-minor-6\.env):

```env
TREEQA_LLM_PROVIDER=openrouter
TREEQA_LLM_BASE_URL=https://openrouter.ai/api/v1
TREEQA_LLM_MODEL=arcee-ai/trinity-mini:free
TREEQA_LLM_FALLBACK_MODELS=nvidia/nemotron-3-nano-30b-a3b:free,arcee-ai/trinity-large-preview:free,qwen/qwen3-4b:free,openrouter/free
OPENROUTER_API_KEY=your_openrouter_key
TREEQA_OPENROUTER_SITE_URL=http://localhost
TREEQA_OPENROUTER_APP_NAME=TreeQA

TREEQA_VECTOR_PROVIDER=local
TREEQA_GRAPH_PROVIDER=local

TREEQA_MAX_RETRIES=2
TREEQA_RETRIEVAL_TOP_K=3
TREEQA_LLM_TIMEOUT_SECONDS=30
TREEQA_LLM_TEMPERATURE=0.0
TREEQA_DATA_DIR=data
```

You can leave the rest blank for now.

Put your source files under:

- [data/documents](d:\CompD\sem6Min\RAG-minor-6\data\documents)
- [data/graph](d:\CompD\sem6Min\RAG-minor-6\data\graph)

Then build the local indices:

```bash
python -m treeqa.cli ingest
```

Before building further, run:

```bash
python -m treeqa.cli doctor
python -m treeqa.cli doctor --live-llm
```

`doctor` validates configuration without network calls. `doctor --live-llm` performs a real LLM probe and will surface issues such as invalid keys, rate limits, or blocked network access.

When multiple fallback models are configured, TreeQA will try them in order until one succeeds.

### When you need more than that

- Add Qdrant when you want real document retrieval over a corpus.
- Add Neo4j when you want structured fact retrieval and graph traversal.
- Use `local` providers first so you can iterate on prompts and workflow behavior with real project documents.
