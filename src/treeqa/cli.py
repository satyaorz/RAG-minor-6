from __future__ import annotations

import json
from dataclasses import asdict

from treeqa.pipeline import TreeQAPipeline


def main() -> None:
    pipeline = TreeQAPipeline()
    query = "How does TreeQA reduce hallucinations and which tools support the workflow?"
    result = pipeline.run(query)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

