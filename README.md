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
3. Run tests: `python -m unittest discover -s tests`
4. Run the demo CLI: `python -m treeqa.cli`
5. Run the UI: `streamlit run src/treeqa/ui/streamlit_app.py`

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

## Validation

The included test suite uses the Python standard library so it runs without extra test tooling.

## Provider setup

Set these env vars to enable real services:

- `TREEQA_LLM_PROVIDER=openai`
- `TREEQA_LLM_MODEL=<model name>`
- `OPENAI_API_KEY=<api key>`
- `TREEQA_LLM_PROVIDER=openrouter`
- `TREEQA_LLM_BASE_URL=https://openrouter.ai/api/v1`
- `TREEQA_LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free`
- `OPENROUTER_API_KEY=<api key>`
- `TREEQA_VECTOR_PROVIDER=qdrant`
- `VECTOR_STORE_URL=<qdrant url>`
- `TREEQA_VECTOR_COLLECTION=<collection>`
- `TREEQA_GRAPH_PROVIDER=neo4j`
- `GRAPH_STORE_URL=<neo4j uri>`

Optional provider packages:

- `pip install -e .[providers]`

### OpenRouter example

Use this configuration to target OpenRouter with the free Llama 3.3 70B instruct model:

```env
TREEQA_LLM_PROVIDER=openrouter
TREEQA_LLM_BASE_URL=https://openrouter.ai/api/v1
TREEQA_LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
OPENROUTER_API_KEY=your-key
TREEQA_OPENROUTER_SITE_URL=http://localhost
TREEQA_OPENROUTER_APP_NAME=TreeQA
```
