from __future__ import annotations

from dataclasses import dataclass, field

from treeqa.models import QueryNode


@dataclass(slots=True)
class WorkflowState:
    query: str
    root: QueryNode
    nodes: list[QueryNode] = field(default_factory=list)
    final_answer: str = ""
