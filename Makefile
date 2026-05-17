PYTHON := .venv/bin/python
OFFLINE_ENV := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TREEQA_EMBEDDING_OFFLINE=1

.PHONY: install install-full doctor doctor-live ingest test bench app streamlit

install:
	$(PYTHON) -m pip install -e ".[api]"

install-full:
	$(PYTHON) -m pip install -e ".[api,providers,datasets,dev]"

doctor:
	$(OFFLINE_ENV) $(PYTHON) -m treeqa.cli doctor

doctor-live:
	$(OFFLINE_ENV) $(PYTHON) -m treeqa.cli doctor --live-llm

ingest:
	$(PYTHON) -m treeqa.cli ingest

test:
	$(OFFLINE_ENV) TREEQA_LLM_PROVIDER=stub $(PYTHON) -m pytest tests/ -q

bench:
	$(OFFLINE_ENV) TREEQA_LLM_PROVIDER=stub $(PYTHON) -m treeqa.cli bench --dataset data/benchmark/sample.jsonl --limit 50 --mode both --warmup 2 --repeats 3

app:
	$(PYTHON) -m uvicorn treeqa.api.app:app --reload --port 8000

streamlit:
	$(PYTHON) -m streamlit run src/treeqa/ui/streamlit_app.py
