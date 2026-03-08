from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievedDocument:
    source_id: str
    source_type: str
    content: str
    score: float


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    confidence: float
    rationale: str


@dataclass(slots=True)
class QueryNode:
    node_id: str
    question: str
    answer: str = ""
    status: str = "pending"
    attempts: int = 0
    documents: list[RetrievedDocument] = field(default_factory=list)
    validation: ValidationResult | None = None
    children: list["QueryNode"] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    query: str
    nodes: list[QueryNode]
    final_answer: str

