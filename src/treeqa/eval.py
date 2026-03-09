"""
treeqa.eval — Local benchmark runner for TreeQA.

Usage
-----
Run against a JSONL file of {question, answer} pairs:

    python -m treeqa.eval --dataset data/benchmark/sample.jsonl

Each input record must have:
  - "question" (str): the query to evaluate
  - "answer"   (str): the expected gold answer

Each output record will have:
  - "question"  (str)
  - "gold"      (str): the gold answer
  - "predicted" (str): TreeQA's final answer
  - "em"        (0 or 1): exact match after normalization
  - "f1"        (float): token F1 between predicted and gold

The runner prints a summary table and writes results to:
  data/benchmark/results_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from treeqa.pipeline import TreeQAPipeline


# ---------------------------------------------------------------------------
# Text normalization helpers (follows HotpotQA evaluation convention)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize_answer(text).split()


def _exact_match(gold: str, predicted: str) -> int:
    return int(_normalize_answer(gold) == _normalize_answer(predicted))


def _token_f1(gold: str, predicted: str) -> float:
    gold_tokens = _tokenize(gold)
    pred_tokens = _tokenize(predicted)
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    common = sum(min(gold_counts.get(t, 0), pred_counts.get(t, 0)) for t in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "question" not in record or "answer" not in record:
            continue
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: Path,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    pipeline = TreeQAPipeline()
    records = _load_dataset(dataset_path)
    if limit is not None:
        records = records[:limit]

    results: list[dict[str, Any]] = []
    total_em = 0
    total_f1 = 0.0

    for i, record in enumerate(records, start=1):
        question = str(record["question"])
        gold = str(record["answer"])
        print(f"[{i}/{len(records)}] {question[:80]}", flush=True)
        try:
            pipeline_result = pipeline.run(question)
            predicted = pipeline_result.final_answer or ""
        except Exception as exc:
            predicted = f"[ERROR: {exc}]"

        em = _exact_match(gold, predicted)
        f1 = _token_f1(gold, predicted)
        total_em += em
        total_f1 += f1

        results.append(
            {
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "em": em,
                "f1": round(f1, 4),
            }
        )

    n = len(results)
    summary = {
        "n": n,
        "exact_match": round(total_em / n, 4) if n else 0.0,
        "f1": round(total_f1 / n, 4) if n else 0.0,
    }

    # Write results
    if output_dir is None:
        output_dir = dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    results_path = output_dir / f"results_{timestamp}.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n--- Benchmark Summary ---")
    print(f"  Questions:   {summary['n']}")
    print(f"  Exact Match: {summary['exact_match']:.1%}")
    print(f"  Token F1:    {summary['f1']:.1%}")
    print(f"  Results:     {results_path}")

    return {**summary, "results_path": str(results_path)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TreeQA local benchmark runner")
    parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark file")
    parser.add_argument("--output-dir", default=None, help="Directory to write result files")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    run_benchmark(dataset_path, output_dir=output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
