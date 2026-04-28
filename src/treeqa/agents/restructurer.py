from __future__ import annotations

from treeqa.backends.llm import LLMClient
from treeqa.models import QueryNode

class TreeRestructurer:
    """
    The 'Self-Healing' component of ART-R.
    When a node fails validation repeatedly, this agent analyzes the failure
    and proposes a structural mutation (new sub-questions).
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def restructure(self, node: QueryNode, rationale: str) -> list[QueryNode] | None:
        """
        Analyzes why a node failed and generates 1-3 new sub-nodes to replace it.
        """
        if self.llm_client is None:
            return None
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a Reasoning Architect. A sub-question in a logic tree has failed "
                    "because the retrieved evidence was insufficient or contradictory.\n"
                    "Your task: Break this sub-question down into even smaller, more verifiable pieces "
                    "OR suggest a different search angle.\n"
                    "Return ONLY a JSON list of objects, each with a 'question' key."
                ),
                user_prompt=(
                    f"Failed Question: {node.question}\n"
                    f"Failure Rationale: {rationale}\n\n"
                    "Provide a more granular decomposition to resolve this logic gap."
                )
            )
            
            if not isinstance(payload, list):
                return None
                
            new_nodes = []
            for i, item in enumerate(payload):
                q = item.get("question")
                if q:
                    new_nodes.append(QueryNode(
                        node_id=f"{node.node_id}.r{i+1}",
                        question=q
                    ))
            return new_nodes if new_nodes else None
            
        except Exception:
            return None
