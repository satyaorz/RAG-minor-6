# PERFORMANCE: Speed & Latency Optimizations

## 1. Overview
As HAMH-RAG (and particularly the ART-R architecture) relies heavily on iterative LLM reasoning and multi-backend data retrieval, system latency is a critical bottleneck. A deep, multi-hop logic tree that triggers self-correction loops could theoretically stack network latencies indefinitely. 

To combat this, the system incorporates targeted concurrency and deterministic caching mechanisms.

## 2. Parallel Hybrid Retrieval
### The Problem:
`HybridRetriever` connects to both an unstructured Vector DB (for semantic similarity) and a structured Knowledge Graph (for factual traversal). Originally, these requests were executed sequentially: `vector_backend.search()` followed by `graph_backend.search()`.

### The Solution:
Introduced `concurrent.futures.ThreadPoolExecutor` inside `HybridRetriever.retrieve()`.
- **Mechanism:** Vector and Graph queries are dispatched simultaneously as separate threads.
- **Impact:** Retrieval latency is no longer the sum of both lookups (`Latency = Vector_Time + Graph_Time`), but rather the maximum of the two (`Latency = max(Vector_Time, Graph_Time)`). For external databases (like Qdrant + Neo4j), this cuts retrieval latency by nearly 50%.

## 3. LLM Network Caching
### The Problem:
During the ART-R "self-healing" loop, the system might retry a branch or validate unchanged nodes multiple times. Executing redundant HTTP requests to an LLM provider (e.g., OpenRouter, OpenAI) adds 1–3 seconds of latency per call.

### The Solution:
Implemented an in-memory execution cache (`_cache`) natively within the `OpenAICompatibleLLMClient`.
- **Mechanism:** Before firing an outbound `urllib.request.urlopen` POST request, the client computes a hash key based on the tuple `(system_prompt, user_prompt)`. If identical, it returns the cached completion.
- **Impact:** Completely eliminates duplicate latency for:
  - Repeated Validator checks on unchanged document contexts.
  - Re-running the Decomposer on parent nodes during whole-tree retries.
  - Generating final syntheses when intermediate nodes haven't fundamentally altered the evidence payload.

## 5. Index & Data Caching
### The Problem:
Benchmark datasets were being streamed and processed from HuggingFace Hub on every load request, and the entire search index (including FAISS embeddings) was being re-computed from scratch during every ingestion, even if no documents had changed.

### The Solution:
Implemented state-aware caching in `dataset_loader.py` and `ingest.py`.
- **Dataset Memoization:** Before downloading, the system checks if the target dataset directory already contains processed documents and a benchmark sample. If present, it skips the network-intensive download and extraction phase entirely.
- **Incremental Index Validation:** The `ingest` module now compares the modification times (`mtime`) of the raw source documents against the existing FAISS index file. 
- **Impact:** 
  - Loading a previously used benchmark dataset is now instantaneous (0s vs 2-5 minutes).
  - Ingestion skips the heavy LLM embedding phase (encoding 20k+ chunks) if the source files are unchanged, reducing server startup and ingestion time from minutes to milliseconds.
