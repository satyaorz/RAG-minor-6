from __future__ import annotations

import re
from typing import Any

from hamhrag.backends.llm import LLMClient
from hamhrag.models import QueryNode


class QueryDecomposer:
    """
    Multi-hop query decomposer.

    Decomposition priority:
      1. Structural heuristics  — dataset-agnostic pattern matching
      2. LLM planner            — handles novel phrasings
      3. Surface-level fallback — conjunctive split on "and"

    Design principle: **structural over keyword**.  We detect 2-hop
    questions by identifying their STRUCTURE (e.g. "attribute of role of
    entity"), not by enumerating predicate verbs.  This makes the
    heuristics robust across HotPotQA, 2WikiMultiHopQA, and MuSiQue.
    """

    # ------------------------------------------------------------------
    # Role vocabulary — roles that introduce a bridge entity
    # ------------------------------------------------------------------
    _ROLES: tuple[str, ...] = (
        "director", "writer", "screenwriter", "author", "producer",
        "composer", "actor", "actress", "creator", "founder", "singer",
        "singer-songwriter", "narrator", "cinematographer", "editor",
        "lyricist", "playwright", "architect", "painter", "photographer",
        "journalist", "pilot", "inventor", "chairman", "president", "ceo",
        "coach", "manager", "captain", "principal", "dean", "governor",
        "senator", "mayor", "officer", "general", "admiral", "professor",
        "teacher", "father", "mother", "brother", "sister", "husband",
        "wife", "spouse", "person", "record label"
    )

    # Pre-compiled role alternation for faster matching
    _ROLE_ALT: str = "|".join(_ROLES)
    _ROLE_PAT: re.Pattern = re.compile(
        r"\b(?:the|a)\s+(?:" + _ROLE_ALT + r")\b",
        re.IGNORECASE,
    )

    # Entity-introducing prepositions / articles in front of entity names
    _ENTITY_INTRO: str = (
        r"(?:the\s+film\s+|film\s+|the\s+movie\s+|movie\s+|"
        r"the\s+book\s+|book\s+|the\s+novel\s+|novel\s+|"
        r"the\s+show\s+|the\s+series\s+|the\s+album\s+|album\s+|"
        r"the\s+song\s+|song\s+|the\s+episode\s+|the\s+play\s+|"
        r"the\s+game\s+|game\s+|the\s+team\s+|the\s+band\s+|"
        r"the\s+poetry\s+collection\s+)?"
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Heuristic dispatcher
    # ------------------------------------------------------------------

    def _heuristic_decompose(self, query: str) -> list[str]:
        """
        Try each structural pattern in priority order.
        Returns [] if no pattern fires (caller falls through to LLM).
        """
        # --- 1. Body-of-water + city border (HotPotQA) ---
        result = self._decompose_body_of_water_bridge(query)
        if result:
            return result

        # --- 2. Event-winner + attribute (HotPotQA / MuSiQue) ---
        result = self._decompose_event_winner_bridge(query)
        if result:
            return result

        # --- 3. STRUCTURAL: "Wh … the ROLE of ENTITY …" ---
        #     Covers ALL attribute-of-person questions regardless of verb.
        #     HotPotQA: "Where did the director of X die?"
        #     2Wiki:    "What is the nationality of the director of X?"
        #     MuSiQue:  "In what city was the founder of X born?"
        result = self._decompose_role_of_entity_bridge(query)
        if result:
            return result

        # --- 3b. STRUCTURAL: Relative clause bridge ---
        result = self._decompose_relative_clause_bridge(query)
        if result:
            return result

        # --- 4. STRUCTURAL: comparison of two entities (A or B?) ---
        #     "Which film has the director who was born earlier, People To Each Other or Tali-Ihantala 1944?"
        #     "Which person lived longer, A or B?"
        result = self._decompose_or_comparison(query)
        if result:
            return result

        # --- 5. STRUCTURAL: same-attribute comparison ---
        #     "Were A and B born in the same country?" (2Wiki)
        #     "Did the director of X and the director of Y share the same nationality?"
        result = self._decompose_same_attribute_comparison(query)
        if result:
            return result

        # --- 6. Explicit conjunction bridge ---
        #     "Who is [person] and where did they [verb]?" (MuSiQue)
        result = self._decompose_explicit_conjunction(query)
        if result:
            return result

        return []

    # ------------------------------------------------------------------
    # Pattern 1: Body-of-water border (HotPotQA)
    # ------------------------------------------------------------------

    def _decompose_body_of_water_bridge(self, query: str) -> list[str]:
        """
        "Which Russian city borders the body of water where X is located?"
        → "Which body of water is X located in?" + "Which city borders that body of water?"
        """
        lowered = query.lower()
        if "body of water" not in lowered or "border" not in lowered:
            return []

        entity: str | None = None
        for pattern in (
            r"body of water (?:in|where) which (?P<entity>[^?]+?) is located",
            r"body of water (?:in|where) (?P<entity>[^?]+?) is located",
        ):
            m = re.search(pattern, query, flags=re.IGNORECASE)
            if m:
                entity = m.group("entity").strip(" ?.;,")
                break
        if not entity:
            return []

        city_clause = "city"
        intro = re.match(
            r"\s*Which\s+(?P<clause>.+?)\s+borders\s+the\s+body\s+of\s+water",
            query,
            flags=re.IGNORECASE,
        )
        if intro:
            clause = intro.group("clause").strip(" ?.;,")
            if clause:
                city_clause = clause

        return [
            f"Which body of water is {entity} located in?",
            f"Which {city_clause} borders that body of water?",
        ]

    # ------------------------------------------------------------------
    # Pattern 2: Event-winner bridge (HotPotQA / MuSiQue)
    # ------------------------------------------------------------------

    def _decompose_event_winner_bridge(self, query: str) -> list[str]:
        """
        "When was the team that won the 1985 World Series established?"
        → "Which team won the 1985 World Series?" + "When was that team established?"

        Generalised beyond baseball: covers any "entity that won/achieved EVENT"
        """
        lowered = query.lower()

        # World Series specific (very common in HotPotQA)
        for pat in (
            r"\b(?:world series in|in)\s+((?:19|20)\d{2})\b",
            r"\b((?:19|20)\d{2})\s+world series\b",
        ):
            m = re.search(pat, lowered)
            if m and "world series" in lowered and "team" in lowered:
                year = m.group(1)
                subject = "baseball team" if "baseball team" in lowered else "team"
                attr = self._extract_attribute_verb(lowered) or "established"
                q2 = f"When was that {subject} {attr}?"
                return [f"Which baseball team won the {year} World Series?", q2]

        # General: "the [entity] that [won/achieved/received] [EVENT]"
        # e.g. "In what year was the film that won Best Picture in 1994 released?"
        m = re.match(
            r"^(.*?)\bthe\s+([\w\s]{1,25}?)\s+that\s+(won|received|achieved|earned|"
            r"claimed|captured|clinched|gained|took)\s+(.+?)\s+(established|founded|"
            r"created|born|released|made|built|started|formed|incorporated|opened)\b",
            lowered,
        )
        if m:
            subject = m.group(2).strip()
            event = m.group(4).strip()
            attr_verb = m.group(5).strip()
            wh = "when" if attr_verb in ("born", "established", "founded", "created",
                                          "released", "made", "built", "started") else "where"
            return [
                f"Which {subject} {m.group(3)} {event}?",
                f"{wh.capitalize()} was that {subject} {attr_verb}?",
            ]

        return []

    @staticmethod
    def _extract_attribute_verb(lowered: str) -> str:
        for v in ("established", "founded", "created", "formed", "incorporated",
                  "started", "built", "opened", "organised", "organized"):
            if v in lowered:
                return v
        return "established"

    # ------------------------------------------------------------------
    # Pattern 3: STRUCTURAL role-of-entity bridge
    # ------------------------------------------------------------------

    def _decompose_role_of_entity_bridge(self, query: str) -> list[str]:
        """
        STRUCTURAL detection — does NOT enumerate predicate verbs.

        Triggers on any question that contains "the ROLE of/in/for ENTITY" where
        the question is NOT merely asking *who* that role is (single hop).
        """
        # Step 1: find "the ROLE of/in/for" anchor
        role_anchor = re.compile(
            r"\b(?:the|a)\s+(" + self._ROLE_ALT + r")\s+(of|in|for)\s+" + self._ENTITY_INTRO,
            re.IGNORECASE,
        )
        m = role_anchor.search(query)
        if not m:
            return []

        role = m.group(1).lower()
        prep = m.group(2).lower()
        entity_start = m.end()

        # Step 2: extract entity — stop at predicate boundary signals:
        #   • a finite verb that begins the predicate (die, was born, live…)
        #   • a relative clause marker (who, which, that, where, when)
        #   • a comma or semicolon
        #   • end of string
        rest = query[entity_start:]

        _PRED_VERBS = re.compile(
            r"(?<!\w)("
            r"die[ds]?|died|born|live[sd]?|lived|work[sed]*|worked|"
            r"come|came|go|went|study|studied|graduate[sd]?|attend[ed]*|"
            r"direct[ed]*|appear[ed]*|became?|marri[ed]*|retir[ed]*|"
            r"buried|grew?|publish[ed]*|wrot[e]?|wrote|start[ed]*|"
            r"begin|began|end[ed]*|serv[ed]*|fight|fought|play[ed]*|"
            r"perform[ed]*|star[red]*|produc[ed]*|releas[ed]*|"
            r"record[ed]*|compos[ed]*|invent[ed]*|discover[ed]*|"
            r"found[ed]*|establish[ed]*|creat[ed]*|educat[ed]*|"
            r"resign[ed]*|pass[ed]*|drown[ed]*|kill[ed]*|win|won|"
            r"reign[ed]*|rule[sd]?|govern[ed]*|lead|led|teach|taught|"
            r"publish[ed]*|translat[ed]*|compil[ed]*|design[ed]*|built?|built"
            r")",
            re.IGNORECASE,
        )

        # Find the first valid predicate verb boundary
        # NOTE: use pv_m NOT m — we must not shadow the role_anchor match `m`
        pv_pos: int | None = None
        for pv_m in _PRED_VERBS.finditer(rest):
            ps = pv_m.start()
            if ps > 0 and rest[ps - 1] == " ":
                pv_pos = ps
                break

        # Clause markers — but ONLY when NOT inside parentheses
        # e.g. "Buried Treasure (1921 Film) die?" — the "Film" inside () is NOT a clause
        _depth = [0]
        clause_pos: int | None = None
        for i, ch in enumerate(rest):
            if ch == "(":
                _depth[0] += 1
            elif ch == ")":
                _depth[0] = max(0, _depth[0] - 1)
            elif _depth[0] == 0:
                if re.match(r"(?i)(who|which|that|where|when|,|;)\b", rest[i:]):
                    # make sure it's preceded by a non-word or start
                    if i == 0 or not rest[i - 1].isalnum():
                        clause_pos = i
                        break

        boundaries = []
        if pv_pos is not None:
            boundaries.append(pv_pos)
        if clause_pos is not None:
            boundaries.append(clause_pos)

        if boundaries:
            entity_end = min(boundaries)
        else:
            entity_end = len(rest)

        entity_raw = rest[:entity_end].strip(" ?.,;:'\"")

        if not entity_raw:
            return []

        entity = entity_raw

        # Step 3: guard — "Who is the ROLE of X?" → single hop
        single_hop = re.match(
            r"^who\s+(is|was|are|were)\s+(?:the|a)\s+(?:" + self._ROLE_ALT + r")\s+(?:of|in|for)\b",
            query.lower(),
        )
        if single_hop:
            return []

        # Step 4: build Q1 — clean identification question
        q1 = f"Who is the {role} {prep} {entity}?"

        # Step 5: build Q2 — substitute "the ROLE of [ENTITY_INTRO] ENTITY"
        # with "that ROLE" and keep the entire predicate intact.
        prefix = query[: m.start()]
        predicate = query[entity_start + entity_end :]
        q2_raw = f"{prefix}that {role} {predicate}"
        q2 = re.sub(r"\s+", " ", q2_raw).strip(" ?") + "?"
        q2 = q2[0].upper() + q2[1:] if q2 else q2

        # Sanity: q2 must still be a question
        if not re.search(
            r"\b(what|who|where|when|which|how|is|was|were|did|does|do|in|at|from|by)\b",
            q2.lower(),
        ):
            return []

        return [q1, q2]

    # ------------------------------------------------------------------
    # Pattern 3b: STRUCTURAL relative clause bridge
    # ------------------------------------------------------------------

    def _decompose_relative_clause_bridge(self, query: str) -> list[str]:
        """
        Matches "What is the [attribute] of the [role] who [verb] [entity]?"
        e.g., "What is the nationality of the singer-songwriter who wrote the poetry collection Early Work ?"
        e.g., "When did the person who said 'I think, therefore I am' live?"
        """
        # Look for a role followed by "who" or "that", a verb, and an entity
        m = re.search(
            r"\bthe\s+(" + self._ROLE_ALT + r")\s+(who|that)\s+(wrote|directed|designed|founded|created|released|started|invented|said|sang|performed|composed|produced|starred in)\s+(.+?)(?=\s+(?:live|die|was|is|did|does|have|has|play)|$)",
            query,
            re.IGNORECASE
        )
        if m:
            role = m.group(1).lower()
            verb = m.group(3).lower()
            entity = m.group(4).strip(" ?.,;:'\"")
            
            # Reconstruct Q1: identify the person
            q1 = f"Who is the {role} that {verb} {entity}?"
            
            # Reconstruct Q2: ask the original question about "that role"
            prefix = query[:m.start()].strip()
            suffix = query[m.end():].strip(" ?")
            
            # Ensure proper spacing and capitalization
            q2_raw = f"{prefix} that {role} {suffix}".strip()
            q2 = re.sub(r"\s+", " ", q2_raw).strip(" ?") + "?"
            q2 = q2[0].upper() + q2[1:] if q2 else q2
            
            return [q1, q2]

        return []

    # ------------------------------------------------------------------
    # Pattern 4: Comparison ("... A or B?")
    # ------------------------------------------------------------------

    def _decompose_or_comparison(self, query: str) -> list[str]:
        """
        Comparison questions ending in 'A or B?' or 'A and B?' (after out of/between)
        e.g., 'Which film has the director who was born earlier, People To Each Other or Tali-Ihantala 1944?'
        e.g., 'Who was born first out of Michelle Tong and Eugene Mcdowell?'
        e.g., 'Who was born later, Joseph Haboush or Alexander Argüelles?'
        """
        lowered = query.lower()

        # Look for choice split
        m = re.search(
            r"(?:,\s*|\bout\s+of\s+|\bbetween\s+)(.+?)\s+(?:or|and)\s+(.+?)\??\s*$",
            query,
            re.IGNORECASE
        )
        if not m:
            return []

        a = m.group(1).strip()
        b = m.group(2).strip()
        prefix = query[:m.start()].strip()
        prefix_lower = prefix.lower()

        # Check if there is a role
        role_match = re.search(
            r"\bthe\s+(" + self._ROLE_ALT + r")\b",
            prefix,
            re.IGNORECASE,
        )

        wh = "when"
        verb = "born"
        aux = "was"
        if "born" in prefix_lower or "birth" in prefix_lower or "older" in prefix_lower or "younger" in prefix_lower:
            wh = "when"
            verb = "born"
            aux = "was"
        elif "die" in prefix_lower or "death" in prefix_lower or "passed" in prefix_lower:
            wh = "when"
            verb = "die"
            aux = "did"
        elif "release" in prefix_lower or "publish" in prefix_lower or "premier" in prefix_lower:
            wh = "when"
            verb = "released" if "release" in prefix_lower or "premier" in prefix_lower else "published"
            aux = "was"
        elif "found" in prefix_lower or "creat" in prefix_lower or "establish" in prefix_lower:
            wh = "when"
            verb = "founded"
            aux = "was"

        a_entity = self._recover_entity_case(query, a)
        b_entity = self._recover_entity_case(query, b)

        if role_match:
            role = role_match.group(1).lower()
            q1 = f"{wh.capitalize()} {aux} the {role} of {a_entity} {verb}?"
            q2 = f"{wh.capitalize()} {aux} the {role} of {b_entity} {verb}?"
        else:
            q1 = f"{wh.capitalize()} {aux} {a_entity} {verb}?"
            q2 = f"{wh.capitalize()} {aux} {b_entity} {verb}?"
        return [q1, q2]

    # ------------------------------------------------------------------
    # Pattern 5: Same-attribute comparison  (2WikiMultiHopQA)
    # ------------------------------------------------------------------

    def _decompose_same_attribute_comparison(self, query: str) -> list[str]:
        """
        "Were A and B born in the same country?" → parallel hops for A and B.
        "Did the director of X and the director of Y share the same nationality?"
        "Does Mario Beaulieu (Senator) have the same nationality as Ebenezer Porter?"
        """
        lowered = query.lower()

        # --- Sub-case B: dual-film director nationality (evaluated first) ---
        # "Do both the director of X and the director of Y have the same nationality?"
        if "director" in lowered and ("same country" in lowered or "same nationality" in lowered):
            segment = self._extract_dual_titles(query)
            if segment:
                parts = re.split(r"\s+and\s+", segment, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    left, right = (p.strip(" ?.;:") for p in parts)
                    if left and right:
                        attr = "country" if "country" in lowered else "nationality"
                        return [
                            f"What {attr} is the director of {left} from?",
                            f"What {attr} is the director of {right} from?",
                        ]

        # --- Sub-case A: "Were [Person A] and [Person B] born in the same [attr]?" ---
        m = re.match(
            r"^(?:were|did|do|are|is|have)\s+(?:both\s+)?"
            r"(?P<a>.+?)\s+and\s+(?P<b>.+?)\s+"
            r"(?:both\s+)?(?:born|die|live|come|from|have|share|hold|attend|graduate|study|located)\b.*"
            r"\bsame\s+(?P<attr>\w+(?:\s+\w+)?)\b",
            lowered,
        )
        if m:
            a_raw = m.group("a").strip()
            b_raw = m.group("b").strip()
            attr = m.group("attr").strip()
            a_entity = self._recover_entity_case(query, a_raw)
            b_entity = self._recover_entity_case(query, b_raw)

            wh = "where" if attr in ("country", "city", "place", "state", "region",
                                      "town", "county", "province") else "what"
            verb = "born in" if "born" in lowered else ("located in" if "located" in lowered else "from")
            
            # If the entities themselves are roles (director of film X)
            if "director" in a_raw and "director" in b_raw:
                return [
                    f"{wh.capitalize()} is the {a_entity} {verb}?",
                    f"{wh.capitalize()} is the {b_entity} {verb}?",
                ]
            
            return [
                f"{wh.capitalize()} is {a_entity} {verb}?",
                f"{wh.capitalize()} is {b_entity} {verb}?",
            ]

        # --- Sub-case C: "Does [A] have the same [attr] as [B]?" ---
        # e.g. "Does Mario Beaulieu (Senator) have the same nationality as Ebenezer Porter?"
        m_as = re.search(
            r"\b(?:have|share|hold)\s+the\s+same\s+(?P<attr>\w+(?:\s+\w+)?)\s+as\s+(?P<b>.+?)\??\s*$",
            lowered,
        )
        if m_as:
            attr = m_as.group("attr").strip()
            b_raw = m_as.group("b").strip()
            prefix = query[:m_as.start()].strip()
            
            a_match = re.search(
                r"^(?:does|did|do|is|are|were|has|have)\s+(.+?)(?:\s+(?:have|share|hold|be|with))?$",
                prefix,
                re.IGNORECASE,
            )
            if a_match:
                a_raw = a_match.group(1).strip()
                a_entity = self._recover_entity_case(query, a_raw)
                b_entity = self._recover_entity_case(query, b_raw)
                
                wh = "where" if attr in ("country", "city", "place", "state", "region",
                                          "town", "county", "province") else "what"
                return [
                    f"{wh.capitalize()} is the {attr} of {a_entity}?" if attr == "nationality" else f"{wh.capitalize()} did {a_entity} live/born?",
                    f"{wh.capitalize()} is the {attr} of {b_entity}?" if attr == "nationality" else f"{wh.capitalize()} did {b_entity} live/born?",
                ]

        return []

    # ------------------------------------------------------------------
    # Pattern 5: Explicit conjunction (MuSiQue)
    # ------------------------------------------------------------------

    def _decompose_explicit_conjunction(self, query: str) -> list[str]:
        """
        MuSiQue often chains two explicit sub-questions with "and":
        "Who wrote the song X and what album does it appear on?"
        "What is the capital of the country where X was born and how large is it?"

        We only split if both halves are independently answerable questions.
        Guard: do NOT split if there is a "the ROLE of ENTITY" bridge (already
        handled above) or if "both" is present (comparison, not chain).
        """
        lowered = query.lower()
        if "both" in lowered:
            return []
        if self._ROLE_PAT.search(query):
            # Bridge questions are handled structurally above
            return []

        # Look for " and " flanked by question words on both sides
        parts = re.split(r"\s+and\s+", query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            return []

        left, right = parts
        # Both halves must look like questions
        wh_re = re.compile(
            r"\b(what|who|where|when|which|how|is|was|were|did|does|do)\b",
            re.IGNORECASE,
        )
        if not wh_re.search(left) or not wh_re.search(right):
            return []

        # Right half must contain an anaphoric pronoun to be a true chain
        anaphora_re = re.compile(
            r"\b(it|its|they|their|that|this|those|these|the same|them)\b",
            re.IGNORECASE,
        )
        if not anaphora_re.search(right):
            return []

        # Ensure both halves are long enough to be real sub-questions
        if len(left.split()) < 3 or len(right.split()) < 2:
            return []

        left_q = left.strip(" ?") + "?"
        right_q = right.strip(" ?") + "?"
        right_q = right_q[0].upper() + right_q[1:]

        return [left_q, right_q]

    # ------------------------------------------------------------------
    # LLM decomposition
    # ------------------------------------------------------------------

    def _decompose_with_llm(self, query: str) -> list[str]:
        if self.llm_client is None:
            return []
        try:
            payload = self.llm_client.generate_json(
                system_prompt=(
                    "You are a Query Architect. Decompose complex multi-hop questions "
                    "into the SMALLEST NECESSARY set of sub-questions.\n\n"
                    "RULES:\n"
                    "1. Simple single-document questions → return as-is (1 sub-question).\n"
                    "2. Split ONLY when finding Entity A is a prerequisite to searching for Entity B.\n"
                    "3. Multiple attributes of the SAME entity → one combined sub-question.\n"
                    "4. NEVER use placeholder tokens like <director>, <person>. "
                    "Use natural pronouns: 'that director', 'they', 'that person'.\n"
                    "5. Bridge questions ('Where did the director of X die?') ALWAYS need "
                    "TWO sub-questions: first identify the person, then find the attribute.\n"
                    "6. Comparison questions ('Were A and B born in the same country?') → "
                    "TWO parallel sub-questions, one per entity.\n\n"
                    "Return JSON: {\"sub_questions\": [\"...\", \"...\"]}."
                ),
                user_prompt=f"Question: {query}",
            )
        except Exception:
            return []
        return self._extract_questions(payload)

    # ------------------------------------------------------------------
    # Sanitization and extraction helpers
    # ------------------------------------------------------------------

    def _sanitize_questions(self, questions: list[str]) -> list[str]:
        cleaned: list[str] = []
        for q in questions:
            q = re.sub(r"\s+", " ", str(q)).strip()
            if not q or len(q.split()) < 3:
                continue
            if not re.search(
                r"\b(what|who|where|when|which|how|is|are|was|were|have|has|had|do|does|did)\b",
                q.lower(),
            ):
                continue
            if not q.endswith("?"):
                q += "?"
            cleaned.append(q)
        return cleaned

    def _extract_questions(self, payload: dict[str, Any] | list[Any]) -> list[str]:
        if isinstance(payload, list):
            candidates = payload
        else:
            candidates = payload.get("sub_questions") or payload.get("questions") or []
        if not isinstance(candidates, list):
            return []
        return [str(c).strip() for c in candidates if str(c).strip()]

    # ------------------------------------------------------------------
    # Surface-level fallback
    # ------------------------------------------------------------------

    def _split_query(self, query: str) -> list[str]:
        normalized = query.replace(" then ", " and ")
        if re.search(r"\bboth\b", normalized, flags=re.IGNORECASE):
            return [query.strip()]
        segments = [s.strip(" ?.") for s in normalized.split(" and ")]
        cleaned = [s for s in segments if s]
        return cleaned or [query.strip()]

    # ------------------------------------------------------------------
    # Tree builder — ALWAYS preserves original root query
    # ------------------------------------------------------------------

    def _build_tree(self, query: str, parts: list[str], visited: set[str] | None = None) -> QueryNode:
        if visited is None:
            visited = set()

        cleaned = [p for p in parts if p]
        if not cleaned:
            return QueryNode(node_id="root", question=query.strip())

        normalized_query = query.strip().lower()
        if normalized_query in visited:
            return QueryNode(
                node_id="root",
                question=query.strip(),
                children=[QueryNode(node_id=f"node-{i}", question=p) for i, p in enumerate(cleaned, start=1)]
            )
        visited.add(normalized_query)

        if len(cleaned) == 1:
            sub = cleaned[0].strip(" ?")
            orig = query.strip(" ?")
            if sub.lower() == orig.lower() or orig.lower().startswith(sub.lower()[:50]):
                return QueryNode(node_id="root", question=query.strip())
            
            # Recursive decomposition for single child with loop protection
            if sub.lower() not in visited:
                child_node = self.decompose(cleaned[0])
                child_node.node_id = "node-1"
            else:
                child_node = QueryNode(node_id="node-1", question=cleaned[0])
            return QueryNode(
                node_id="root",
                question=query.strip(),
                children=[child_node],
            )

        # Multiple sub-questions: recursively decompose each child if it can be decomposed further
        children = []
        for i, part in enumerate(cleaned, start=1):
            normalized_part = part.strip().lower()
            if normalized_part in visited:
                child_node = QueryNode(node_id=f"node-{i}", question=part)
            else:
                sub_parts = self._heuristic_decompose(part)
                if sub_parts and len(sub_parts) > 1 and not any(sp.strip().lower() == normalized_part for sp in sub_parts):
                    child_node = self._build_tree(part, sub_parts, visited.copy())
                    child_node.node_id = f"node-{i}"
                else:
                    child_node = QueryNode(node_id=f"node-{i}", question=part)
            children.append(child_node)

        return QueryNode(
            node_id="root",
            question=query.strip(),
            children=children,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _recover_entity_case(self, original_query: str, entity_lower: str) -> str:
        """Return the original-cased substring matching entity_lower."""
        start = original_query.lower().find(entity_lower)
        if start >= 0:
            return original_query[start : start + len(entity_lower)].strip(" ?.,;:'\"")
        return entity_lower.strip()

    def _extract_dual_titles(self, query: str) -> str | None:
        m = re.search(
            r"\bboth\s+(?:directors\s+of\s+|writers\s+of\s+|founders\s+of\s+|[a-zA-Z\s]+of\s+)?(?:films?|movies?|titles?|works?|directors|writers|founders|performers)?\s*:?\s*(.+?)\s+(?:have|share|both|are|were|do|did)\b",
            query,
            flags=re.IGNORECASE,
        )
        return m.group(1).strip() if m else None
