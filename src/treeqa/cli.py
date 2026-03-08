from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from treeqa.diagnostics import run_diagnostics
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


if __name__ == "__main__":
    main()
