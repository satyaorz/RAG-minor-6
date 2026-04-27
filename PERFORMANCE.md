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

## 6. Consolidated Logic Auditing
### The Problem:
Deep validation (Grounding + Category + Consensus) previously required 2–3 sequential LLM calls per attempt, causing timeouts (110s+) on free-tier providers.

### The Solution:
Implemented a **Consolidated Logic Auditor** in `validator.py`.
- **Mechanism:** A single unified prompt evaluates evidence grounding, entity category alignment, and source consensus simultaneously.
- **Impact:** Reduces per-node validation latency by 60–70%, ensuring complex multi-hop trees resolve well within the timeout window.

## 7. Intelligent Early-Exit (Short-Circuiting)
### The Problem:
If a validator fails due to a structural issue (e.g., asking for a Country but only finding a Region), retrying the same question with a "refined" query is usually futile.

### The Solution:
Implemented **Categorical Short-Circuiting** in `pipeline.py`.
- **Mechanism:** If the validator returns a `[Category Mismatch]` rationale, the pipeline immediately breaks the retry loop and triggers the **ART-R Restructurer**.
- **Impact:** Prevents 2–4 useless LLM calls per failed hop, significantly accelerating the "self-healing" phase of the logic tree.
