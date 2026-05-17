# HAMH-RAG Benchmark Protocol

## Goals
- Primary: HAMH-RAG should be faster than Normal RAG on the same question set while maintaining reasonable answer quality.
- Target (initial): >= 1.2x throughput vs Normal RAG on the local sample benchmark with warm cache.
- Secondary: P95 latency <= 2x Normal RAG on the same set.

## Benchmark Dataset
- Default: data/benchmark/sample.jsonl
- Fields: question (required), answer (optional for latency-only runs).

## Procedure
1. Build indices (if needed): python -m treeqa.cli ingest
2. Warmup: 2 questions (default)
3. Runs: 1 repeat (default), scale to 3 for stable numbers
4. Modes: hamh and rag
5. Record: avg, p50, p90, p95, p99, throughput (qps)

Benchmarks force the LLM provider to `stub` to avoid heavyweight model imports and to focus on retrieval/runtime overhead.

## Command
- python -m treeqa.cli bench --dataset data/benchmark/sample.jsonl --limit 50 --mode both

## Notes
- Run on the same machine with no other heavy workloads.
- For publishable results, use at least 200 questions and 3 repeats.
