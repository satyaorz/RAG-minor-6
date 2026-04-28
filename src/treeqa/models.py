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
    status: str = "pending"  # pending, verified, needs_review, restructured
    attempts: int = 0
    documents: list[RetrievedDocument] = field(default_factory=list)
    validation: ValidationResult | None = None
    children: list["QueryNode"] = field(default_factory=list)
    hop_context: str = ""
    
    # ART-R specific tracking
    source_consensus: float = 1.0  # 1.0 = total agreement, 0.0 = total conflict
    was_restructured: bool = False
    original_question: str | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def iter_nodes(self) -> list["QueryNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.iter_nodes())
        return nodes


@dataclass(slots=True)
class PipelineResult:
    query: str
    root: QueryNode
    nodes: list[QueryNode]
    final_answer: str
