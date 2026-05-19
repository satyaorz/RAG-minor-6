from hamhrag.models import QueryNode, ValidationResult
import re

def _target_person_attribute_node(query: str, nodes: list[QueryNode]) -> QueryNode | None:
    lowered = query.lower()

    # Detect wh-word of the root question
    wh_match = re.match(r"^(where|when|what|who|which|how)", lowered)
    if not wh_match:
        return None
    wh = wh_match.group(1)

    # STRICTER anaphora detection: only actual pronouns/demonstratives used in hop-2.
    # Exclude "the <role>" because it falsely matches hop-1 (e.g. "Who is the director?").
    anaphora_re = re.compile(
        r"\b(that|their|its|this|they)\b",
        re.IGNORECASE,
    )

    candidates = []
    for node in nodes:
        if not node.answer:
            continue
        node_q_lower = node.question.lower()
        if not anaphora_re.search(node_q_lower):
            continue
        # Bonus: the node's wh-word matches the root's wh-word
        node_wh = re.match(r"^(where|when|what|who|which|how)", node_q_lower)
        score = 2 if (node_wh and node_wh.group(1) == wh) else 1
        candidates.append((score, node))

    if not candidates:
        return None
    # Highest score, last occurrence wins
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

nodes = [
    QueryNode(
        node_id="node-1",
        question="Who is the director of film Blind Spot (1932 Film)?",
        status="verified",
        answer="John Daumery Sources: 1",
        validation=ValidationResult(passed=True, confidence=1.0, rationale="")
    ),
    QueryNode(
        node_id="node-2",
        question="What is that person's mother?",
        status="verified",
        answer="Carrie Daumery Sources: 2",
        validation=ValidationResult(passed=True, confidence=1.0, rationale="")
    )
]

query = "Who is the mother of the director of film Blind Spot (1932 Film)?"
target = _target_person_attribute_node(query, nodes)
print("Target node:", target.question if target else "None")

