from __future__ import annotations

import re
from typing import Any

from treeqa.backends.llm import LLMClient
from treeqa.models import QueryNode


class QueryDecomposer:
    """Uses an LLM planner when configured, with a rule-based fallback."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def decompose(self, query: str) -> QueryNode:
        heuristic_parts = self._heuristic_decompose(query)
        if heuristic_parts:
            return self._build_tree(query, heuristic_parts)
        llm_parts = self._sanitize_questions(self._decompose_with_llm(query))
        if llm_parts:
            return self._build_tree(query, llm_parts)
        parts = self._split_query(query)
        return self._build_tree(query, parts)

    def _split_query(self, query: str) -> list[str]:
        normalized = query.replace(" then ", " and ")
        if re.search(r"\bboth\b", normalized, flags=re.IGNORECASE):
            return [query.strip()]
        segments = [segment.strip(" ?.") for segment in normalized.split(" and ")]
        cleaned = [segment for segment in segments if segment]
        return cleaned or [query.strip()]

    def _decompose_with_llm(self, query: str) -> list[str]:
        if self.llm_client is None:
            return []
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a Query Architect. Your goal is to decompose complex multi-hop questions "
                    "into the SMALLEST NECESSARY set of sub-questions.\n\n"
                    "CRITICAL RULES:\n"
                    "1. If a question is simple and can be answered from a single document, DO NOT SPLIT IT. "
                    "Return it as a single sub-question.\n"
                    "2. Only split if you truly need to find Entity A before you can even search for Entity B.\n"
                    "3. If multiple attributes about the SAME entity are requested (e.g., director + nationality), "
                    "combine them into ONE sub-question.\n"
                    "4. Do not create redundant or 'exploratory' questions (e.g., 'Who is X?').\n\n"
                    "Return JSON: {\"sub_questions\": [\"...\"]}."
                ),
                user_prompt=f"Question: {query}",
            )

        except Exception:
            return []
        return self._extract_questions(payload)

    def _sanitize_questions(self, questions: list[str]) -> list[str]:
        if not questions:
            return []
        cleaned: list[str] = []
        for q in questions:
            q = re.sub(r"\s+", " ", str(q)).strip()
            # Keep short but valid questions (e.g., "Who built TreeQA?").
            if not q or len(q.split()) < 3:
                continue
            if not re.search(
                r"\b(what|who|where|when|which|how|is|are|was|were|have|has|had|do|does|did)\b",
                q.lower(),
            ):
                continue
            if not q.endswith("?"):
                q = q + "?"
            cleaned.append(q)
        return cleaned

    def _extract_questions(self, payload: dict[str, Any] | list[Any]) -> list[str]:
        if isinstance(payload, list):
            candidates = payload
        else:
            candidates = payload.get("sub_questions") or payload.get("questions") or []
        if not isinstance(candidates, list):
            return []
        return [str(candidate).strip() for candidate in candidates if str(candidate).strip()]

    def _heuristic_decompose(self, query: str) -> list[str]:
        lowered = query.lower()
        water_bridge = self._decompose_body_of_water_bridge(query)
        if water_bridge:
            return water_bridge
        creation_bridge = self._decompose_creation_bridge(query)
        if creation_bridge:
            return creation_bridge

        if "director" not in lowered:
            return []
        if "both" not in lowered:
            return []
        if "same country" not in lowered and "same nationality" not in lowered:
            return []

        segment = self._extract_dual_titles(query)
        if not segment:
            return []

        parts = re.split(r"\s+and\s+", segment, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            return []
        left, right = (p.strip(" ?.;:") for p in parts)
        if not left or not right:
            return []

        if "country" in lowered:
            template = "What country is the director of {title} from?"
        else:
            template = "What is the nationality of the director of {title}?"

        return [template.format(title=left), template.format(title=right)]

    def _decompose_creation_bridge(self, query: str) -> list[str]:
        lowered = query.lower()
        if "when" not in lowered:
            return []
        if not re.search(r"\b(created|founded|formed|established|started|inaugurated)\b", lowered):
            return []

        world_series_year = re.search(r"\b(?:world series in|in)\s+((?:19|20)\d{2})\b", lowered)
        if world_series_year and "world series" in lowered and "team" in lowered:
            year = world_series_year.group(1)
            return [
                f"Which baseball team won the {year} World Series?",
                "When was that baseball team established?",
            ]

        year_world_series = re.search(r"\b((?:19|20)\d{2})\s+world series\b", lowered)
        if year_world_series and "team" in lowered:
            year = year_world_series.group(1)
            return [
                f"Which baseball team won the {year} World Series?",
                "When was that baseball team established?",
            ]

        return []

    def _decompose_body_of_water_bridge(self, query: str) -> list[str]:
        lowered = query.lower()
        if "body of water" not in lowered or "border" not in lowered:
            return []

        entity = None
        patterns = [
            r"body of water (?:in|where) which (?P<entity>[^?]+?) is located",
            r"body of water (?:in|where) (?P<entity>[^?]+?) is located",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                entity = match.group("entity").strip(" ?.;,")
                break
        if not entity:
            return []

        city_clause = "city"
        intro_match = re.match(
            r"\s*Which\s+(?P<clause>.+?)\s+borders\s+the\s+body\s+of\s+water",
            query,
            flags=re.IGNORECASE,
        )
        if intro_match:
            clause = intro_match.group("clause").strip(" ?.;,")
            if clause:
                city_clause = clause

        return [
            f"Which body of water is {entity} located in?",
            f"Which {city_clause} borders that body of water?",
        ]

    def _extract_dual_titles(self, query: str) -> str | None:
        match = re.search(
            r"both\s+(?:films|film|movies|movie|titles|works)?\s*:?(.*?)\s+have\b",
            query,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    def _build_tree(self, query: str, parts: list[str]) -> QueryNode:
        cleaned = [part for part in parts if part]
        if len(cleaned) <= 1:
            question = cleaned[0] if cleaned else query.strip()
            return QueryNode(node_id="root", question=question)
        return QueryNode(
            node_id="root",
            question=query.strip(),
            children=[
                QueryNode(node_id=f"node-{index}", question=part)
                for index, part in enumerate(cleaned, start=1)
            ],
        )
