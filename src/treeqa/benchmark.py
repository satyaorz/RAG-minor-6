"""
treeqa.benchmark — Latency/throughput benchmark for HAMH-RAG vs Normal RAG.

Usage
-----
python -m treeqa.benchmark --dataset data/benchmark/sample.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from treeqa.config import TreeQASettings
from treeqa.pipeline import TreeQAPipeline


@dataclass(slots=True)
class BenchmarkSummary:
    mode: str
    n: int
    total_seconds: float
    throughput_qps: float
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(slots=True)
class BenchmarkResult:
    summaries: list[BenchmarkSummary]
    samples: list[dict[str, Any]]
    results_path: str
    comparison: dict[str, float] | None = None


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "question" not in record:
            continue
        records.append(record)
    return records


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = max(0, min(len(values_sorted) - 1, int(round((pct / 100.0) * (len(values_sorted) - 1)))))
    return values_sorted[k]


def _run_normal_rag(pipeline: TreeQAPipeline, question: str) -> None:
    top_k = max(1, int(getattr(pipeline.retriever, "top_k", pipeline.settings.retrieval_top_k)))
    vector_backend = getattr(pipeline.retriever, "vector_backend", None)
    if vector_backend is not None and hasattr(vector_backend, "search"):
        docs = vector_backend.search(question, top_k)
    else:
        docs = []
    _ = pipeline.generator.generate_for_node(question, docs)


def _benchmark_mode(
    pipeline: TreeQAPipeline,
    questions: Iterable[str],
    mode: str,
    warmup: int,
    repeats: int,
) -> tuple[BenchmarkSummary, list[dict[str, Any]]]:
    latencies_ms: list[float] = []
    samples: list[dict[str, Any]] = []

    def run_once(q: str) -> float:
        start = time.perf_counter()
        if mode == "hamh":
            _ = pipeline.run(q)
        else:
            _run_normal_rag(pipeline, q)
        return (time.perf_counter() - start) * 1000.0

    questions_list = list(questions)
    for q in questions_list[:warmup]:
        run_once(q)

    for _ in range(repeats):
        for q in questions_list:
            latency = run_once(q)
            latencies_ms.append(latency)
            samples.append({"mode": mode, "question": q, "latency_ms": round(latency, 2)})

    total_seconds = sum(latencies_ms) / 1000.0
    n = len(latencies_ms)
    summary = BenchmarkSummary(
        mode=mode,
        n=n,
        total_seconds=round(total_seconds, 4),
        throughput_qps=round((n / total_seconds) if total_seconds else 0.0, 4),
        avg_ms=round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        p50_ms=round(_percentile(latencies_ms, 50), 2),
        p90_ms=round(_percentile(latencies_ms, 90), 2),
        p95_ms=round(_percentile(latencies_ms, 95), 2),
        p99_ms=round(_percentile(latencies_ms, 99), 2),
    )
    return summary, samples


def run_benchmark(
    dataset_path: Path,
    output_dir: Path | None = None,
    limit: int | None = None,
    warmup: int = 2,
    repeats: int = 1,
    mode: str = "both",
) -> BenchmarkResult:
    settings = TreeQASettings.from_env()
    settings.llm_provider = "stub"
    settings.embedding_offline = True
    pipeline = TreeQAPipeline(settings=settings)
    records = _load_dataset(dataset_path)
    if limit is not None:
        records = records[:limit]
    questions = [str(record["question"]) for record in records]

    summaries: list[BenchmarkSummary] = []
    samples: list[dict[str, Any]] = []

    if mode in {"hamh", "both"}:
        summary, mode_samples = _benchmark_mode(pipeline, questions, "hamh", warmup, repeats)
        summaries.append(summary)
        samples.extend(mode_samples)

    if mode in {"rag", "both"}:
        summary, mode_samples = _benchmark_mode(pipeline, questions, "rag", warmup, repeats)
        summaries.append(summary)
        samples.extend(mode_samples)

    comparison: dict[str, float] | None = None
    if len(summaries) == 2:
        by_mode = {summary.mode: summary for summary in summaries}
        hamh = by_mode.get("hamh")
        rag = by_mode.get("rag")
        if hamh and rag and rag.throughput_qps > 0:
            comparison = {
                "throughput_ratio_hamh_vs_rag": round(hamh.throughput_qps / rag.throughput_qps, 4),
                "latency_ratio_hamh_vs_rag": round(hamh.avg_ms / rag.avg_ms, 4) if rag.avg_ms else 0.0,
            }

    if output_dir is None:
        output_dir = dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    results_path = output_dir / f"benchmark_{timestamp}.json"

    payload = {
        "dataset": str(dataset_path),
        "mode": mode,
        "warmup": warmup,
        "repeats": repeats,
        "summaries": [asdict(s) for s in summaries],
        "comparison": comparison,
        "samples": samples,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return BenchmarkResult(
        summaries=summaries,
        samples=samples,
        results_path=str(results_path),
        comparison=comparison,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HAMH-RAG latency/throughput benchmark")
    parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark file")
    parser.add_argument("--output-dir", default=None, help="Directory to write result files")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup questions to run before timing")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat full question set N times")
    parser.add_argument(
        "--mode",
        choices=["hamh", "rag", "both"],
        default="both",
        help="Benchmark mode: hamh, rag, or both",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    result = run_benchmark(
        dataset_path,
        output_dir=output_dir,
        limit=args.limit,
        warmup=args.warmup,
        repeats=args.repeats,
        mode=args.mode,
    )

    print("\n--- Benchmark Summary ---")
    for summary in result.summaries:
        print(f"Mode:        {summary.mode}")
        print(f"  Runs:      {summary.n}")
        print(f"  Total:     {summary.total_seconds:.2f}s")
        print(f"  Throughput:{summary.throughput_qps:.2f} qps")
        print(f"  Avg:       {summary.avg_ms:.2f} ms")
        print(f"  P50:       {summary.p50_ms:.2f} ms")
        print(f"  P90:       {summary.p90_ms:.2f} ms")
        print(f"  P95:       {summary.p95_ms:.2f} ms")
        print(f"  P99:       {summary.p99_ms:.2f} ms")
    if result.comparison:
        print("Comparison:")
        print(f"  Throughput ratio (HAMH/RAG): {result.comparison['throughput_ratio_hamh_vs_rag']:.2f}x")
        print(f"  Latency ratio (HAMH/RAG):    {result.comparison['latency_ratio_hamh_vs_rag']:.2f}x")
    print(f"Results: {result.results_path}")


if __name__ == "__main__":
    main()
