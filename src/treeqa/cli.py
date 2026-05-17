from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from treeqa.diagnostics import run_diagnostics
from treeqa.benchmark import run_benchmark
from treeqa.ingest import build_local_indices
from treeqa.pipeline import TreeQAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="treeqa")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the TreeQA pipeline")
    run_parser.add_argument(
        "query",
        nargs="?",
        default="How does TreeQA reduce hallucinations and which tools support the workflow?",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Inspect provider and config health")
    doctor_parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Perform a live LLM probe instead of config-only validation.",
    )
    subparsers.add_parser("ingest", help="Build local retrieval indices from data/")

    bench_parser = subparsers.add_parser("bench", help="Benchmark HAMH-RAG latency/throughput")
    bench_parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark file")
    bench_parser.add_argument("--output-dir", default=None, help="Directory to write result files")
    bench_parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    bench_parser.add_argument("--warmup", type=int, default=2, help="Warmup questions to run before timing")
    bench_parser.add_argument("--repeats", type=int, default=1, help="Repeat full question set N times")
    bench_parser.add_argument(
        "--mode",
        choices=["hamh", "rag", "both"],
        default="both",
        help="Benchmark mode: hamh, rag, or both",
    )

    args = parser.parse_args()
    if args.command in {None, "run"}:
        pipeline = TreeQAPipeline()
        result = pipeline.run(args.query)
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "doctor":
        report = run_diagnostics(live_llm_probe=args.live_llm)
        print(json.dumps(report.to_dict(), indent=2))
        return

    if args.command == "ingest":
        report = build_local_indices()
        print(json.dumps(asdict(report), indent=2))
        return

    if args.command == "bench":
        output_dir = Path(args.output_dir) if args.output_dir else None
        result = run_benchmark(
            Path(args.dataset),
            output_dir=output_dir,
            limit=args.limit,
            warmup=args.warmup,
            repeats=args.repeats,
            mode=args.mode,
        )
        print(json.dumps({
            "summaries": [asdict(s) for s in result.summaries],
            "results_path": result.results_path,
            "comparison": result.comparison,
        }, indent=2))


if __name__ == "__main__":
    main()
