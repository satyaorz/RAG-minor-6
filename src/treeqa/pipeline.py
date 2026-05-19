from __future__ import annotations

import re
import time
from typing import Callable

from treeqa.agents import (
    AnswerGenerator,
    AnswerValidator,
    ConflictAuditor,
    CorrectionEngine,
    QueryDecomposer,
    TreeRestructurer,
)
from treeqa.backends import build_graph_backend, build_llm_client, build_vector_backend
from treeqa.config import TreeQASettings
from treeqa.models import PipelineResult, QueryNode, RetrievedDocument, ValidationResult
from treeqa.retrieval import HybridRetriever
from treeqa.retrieval.scoring import rank_documents
from treeqa.state import WorkflowState

# Callable[[str, **Any], None] — pipeline events are keyword-only
ProgressCallback = Callable[..., None]


def _noop(**_kwargs) -> None:
    pass


class TreeQAPipeline:
    def __init__(
        self,
        settings: TreeQASettings | None = None,
        decomposer: QueryDecomposer | None = None,
        retriever: HybridRetriever | None = None,
        validator: AnswerValidator | None = None,
        corrector: CorrectionEngine | None = None,
        generator: AnswerGenerator | None = None,
        restructurer: TreeRestructurer | None = None,
        conflict_auditor: ConflictAuditor | None = None,
    ) -> None:
        self.settings = settings or TreeQASettings.from_env()
        llm_client = build_llm_client(self.settings)
        self.decomposer = decomposer or QueryDecomposer(llm_client=llm_client)
        self.retriever = retriever or HybridRetriever(
            vector_backend=build_vector_backend(self.settings),
            graph_backend=build_graph_backend(self.settings),
            top_k=self.settings.retrieval_top_k,
        )
        self.validator = validator or AnswerValidator(llm_client=llm_client)
        self.corrector = corrector or CorrectionEngine(llm_client=llm_client)
        self.generator = generator or AnswerGenerator(llm_client=llm_client)
        self.restructurer = restructurer or TreeRestructurer(llm_client=llm_client)
        self.conflict_auditor = conflict_auditor or ConflictAuditor(llm_client=llm_client)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        progress_callback: ProgressCallback | None = None,
        max_seconds: float | None = None,
        max_retries_override: int | None = None,
        tree_retries_override: int | None = None,
        max_restructures_override: int | None = None,
    ) -> PipelineResult:
        """Run HAMH-RAG on *query*, optionally streaming progress events via *progress_callback*.

        Tree-level retry: if the fully resolved tree status is ``needs_review``
        and ``settings.tree_retries > 0``, the query is refined by the corrector,
        the decomposition tree is re-built, and the pipeline re-runs up to
        ``tree_retries`` additional times.

        *progress_callback* is called with keyword arguments:
            event (str)    — one of start / tree_attempt / decomposed /
                             node_start / node_done / tree_retry / done
            plus event-specific fields (question, nodes, status, answer, …)
        """
        _cb: ProgressCallback = progress_callback or _noop

        _cb(event="start", query=query)

        active_query = query
        result: PipelineResult | None = None
        max_retries = (
            self.settings.max_retries
            if max_retries_override is None
            else max(0, max_retries_override)
        )
        tree_retries = (
            self.settings.tree_retries
            if tree_retries_override is None
            else max(0, tree_retries_override)
        )
        max_restructures = (
            self.settings.max_restructures
            if max_restructures_override is None
            else max(0, max_restructures_override)
        )

        deadline = time.monotonic() + max_seconds if max_seconds else None

        for tree_attempt in range(tree_retries + 1):
            if tree_attempt > 0:
                # Refine the root query and rebuild the decomposition tree
                active_query = self.corrector.refine(active_query, tree_attempt)
                _cb(event="tree_retry", attempt=tree_attempt, refined_query=active_query)

            _cb(event="tree_attempt", attempt=tree_attempt + 1,
                total_attempts=tree_retries + 1)

            root = self.decomposer.decompose(active_query)
            # Carry over verified answers from the previous attempt so we
            # don't re-run nodes that are structurally identical across retries.
            prev_answers: dict[tuple[str, str], str] = (
                {
                    (n.question, getattr(n, "hop_context", "")): n.answer
                    for n in result.nodes
                    if n.status == "verified" and n.answer
                }
                if result is not None else {}
            )
            state = WorkflowState(
                query=active_query, root=root, nodes=self._flatten_leaves(root)
            )
            _cb(event="decomposed",
                nodes=[n.question for n in state.nodes],
                total=len(state.nodes))

            self._resolve_tree(
                state.root,
                progress_callback=_cb,
                verified_cache=prev_answers,
                deadline=deadline,
                max_retries=max_retries,
                max_restructures=max_restructures,
            )
            state.nodes = self._flatten_leaves(state.root)
            verified_nodes = [n for n in state.nodes if n.status == "verified"]
            if state.root.status == "verified" and state.root.answer:
                state.final_answer = state.root.answer
            else:
                state.final_answer = self.generator.generate_final(state.query, verified_nodes)

            result = PipelineResult(
                query=state.query,
                root=state.root,
                nodes=state.nodes,
                final_answer=state.final_answer,
            )

            # Exit early if the tree was fully verified
            if state.root.status == "verified":
                break

        _cb(event="done", answer=result.final_answer, status=result.root.status)  # type: ignore[union-attr]
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tree / leaf resolution
    # ------------------------------------------------------------------

    def _resolve_tree(
        self,
        node: QueryNode,
        prior_hops: list[tuple[str, str]] | None = None,
        progress_callback: ProgressCallback | None = None,
        verified_cache: dict[tuple[str, str], str] | None = None,
        restructure_depth: int = 0,
        deadline: float | None = None,
        max_retries: int | None = None,
        max_restructures: int | None = None,
    ) -> None:
        _cb: ProgressCallback = progress_callback or _noop
        if prior_hops is None:
            prior_hops = []
        if verified_cache is None:
            verified_cache = {}

        if deadline is not None and time.monotonic() > deadline:
            self._mark_timeout(node)
            return
            
        if node.children:
            import concurrent.futures
            
            # Parallelization is powerful but dangerous for multi-hop chaining.
            # We ONLY parallelize if:
            # 1. The question is a comparison ("which", "compare", "difference")
            # 2. OR the decomposer explicitly tags nodes as independent (future-proof)
            # Default to SEQUENTIAL for safety (fixes multi-hop dependencies).
            
            is_comparison = any(
                k in node.question.lower() for k in ["which", "compare", "difference", "between"]
            )
            is_tagged_parallel = "[PARALLEL]" in node.question
            
            has_dependent_children = any(
                "[DEP:" in child.question for child in node.children
            ) or any(
                self._question_uses_anaphora(child.question)
                for child in node.children[1:]
            )
            do_parallel = (is_comparison or is_tagged_parallel) and not has_dependent_children
            
            if do_parallel:
                if deadline is not None and time.monotonic() > deadline:
                    self._mark_timeout(node)
                    return
                # Independent siblings can run in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(node.children)) as executor:
                    futures = {
                        executor.submit(
                            self._resolve_tree, 
                            child, 
                            prior_hops, 
                            _cb, 
                            verified_cache, 
                            restructure_depth,
                            deadline,
                            max_retries,
                            max_restructures,
                        ): child for child in node.children
                    }
                    concurrent.futures.wait(futures)
            else:
                # Sequential resolution for dependent hops
                accumulated: list[tuple[str, str]] = list(prior_hops)
                propagate_sibling_context = has_dependent_children
                for child in node.children:
                    if deadline is not None and time.monotonic() > deadline:
                        self._mark_timeout(node)
                        return
                    hop_input = accumulated if propagate_sibling_context else list(prior_hops)
                    self._resolve_tree(
                        child, 
                        hop_input, 
                        _cb, 
                        verified_cache, 
                        restructure_depth,
                        deadline,
                        max_retries,
                        max_restructures,
                    )
                    if propagate_sibling_context and child.status == "verified" and child.answer:
                        accumulated.append((child.question, self._strip_sources(child.answer)))
            
            verified_children = [child for child in node.children if child.status == "verified"]
            node.answer = self.generator.generate_final(node.question, verified_children)
            node.attempts = max((child.attempts for child in node.children), default=0)
            node.status = (
                "verified"
                if all(child.status == "verified" for child in node.children)
                else "needs_review"
            )
            confidences = [
                child.validation.confidence
                for child in node.children
                if child.validation is not None
            ]
            if confidences:
                node.validation = self._build_group_validation(node.status, confidences)
            consensus_scores = [
                child.source_consensus
                for child in node.children
                if isinstance(child.source_consensus, float)
            ]
            if consensus_scores:
                node.source_consensus = sum(consensus_scores) / len(consensus_scores)
            return
        node.hop_context = self._hop_context(prior_hops)
        cache_key = self._cache_key(node.question, prior_hops)
        if cache_key in verified_cache:
            node.answer = verified_cache[cache_key]
            node.status = "verified"
            node.attempts = 0
            _cb(event="node_start", node_id=node.node_id, question=node.question)
            _cb(event="node_done", node_id=node.node_id, question=node.question,
                status="verified", answer=self._strip_sources(node.answer), attempts=0)
            return
        self._resolve_leaf(
            node, 
            prior_hops, 
            progress_callback=_cb, 
            verified_cache=verified_cache,
            restructure_depth=restructure_depth,
            deadline=deadline,
            max_retries=max_retries,
            max_restructures=max_restructures,
        )

    def _resolve_leaf(
        self,
        node: QueryNode,
        prior_hops: list[tuple[str, str]] | None = None,
        progress_callback: ProgressCallback | None = None,
        verified_cache: dict[tuple[str, str], str] | None = None,
        restructure_depth: int = 0,
        deadline: float | None = None,
        max_retries: int | None = None,
        max_restructures: int | None = None,
    ) -> None:
        _cb: ProgressCallback = progress_callback or _noop
        prior_hops = prior_hops or []
        base_question = node.question
        node.hop_context = self._hop_context(prior_hops)
        contextual_question = self._contextualize_question(base_question, prior_hops)
        retrieval_question = self._enrich_query(contextual_question, prior_hops)

        if deadline is not None and time.monotonic() > deadline:
            self._mark_timeout(node)
            return

        _cb(event="node_start", node_id=node.node_id, question=base_question)
        retry_cap = self.settings.max_retries if max_retries is None else max(0, max_retries)
        restructure_cap = (
            self.settings.max_restructures
            if max_restructures is None
            else max(0, max_restructures)
        )
        skip_restructure = False
        conflict_attempted = False

        for attempt in range(retry_cap + 1):
            if deadline is not None and time.monotonic() > deadline:
                self._mark_timeout(node)
                return
            node.attempts = attempt + 1
            retrieval_trace = None
            retrieve_with_trace = getattr(self.retriever, "retrieve_with_trace", None)
            has_native_trace = callable(retrieve_with_trace) and hasattr(type(self.retriever), "retrieve_with_trace")
            if has_native_trace:
                node.documents, retrieval_trace = retrieve_with_trace(retrieval_question)
            else:
                node.documents = self.retriever.retrieve(retrieval_question)
            if not node.documents:
                node.answer = f"Insufficient evidence to answer: {base_question}"
                node.validation = ValidationResult(
                    passed=False,
                    confidence=0.0,
                    rationale="No evidence found for this sub-question.",
                )
                node.status = "needs_review"
                skip_restructure = True
                break
            if retrieval_trace is not None:
                node.retrieval_route = retrieval_trace.route
                node.retrieval_reason = retrieval_trace.reason
                node.retrieval_backends = list(retrieval_trace.backends_used)
                node.retrieval_latency_ms = retrieval_trace.total_latency_ms
                node.retrieval_fallback = retrieval_trace.fallback_used
                node.retrieval_signals = dict(retrieval_trace.signals)
                _cb(
                    event="node_route",
                    node_id=node.node_id,
                    route=node.retrieval_route,
                    reason=node.retrieval_reason,
                    backends=node.retrieval_backends,
                    fallback=node.retrieval_fallback,
                    latency_ms=round(node.retrieval_latency_ms, 2),
                )
            bridge_query, bridge_docs = self._bridge_director_country_evidence(
                contextual_question, node.documents
            )
            if bridge_docs:
                merge_limit = max(
                    4,
                    self.settings.retrieval_top_k + 1,
                    int(getattr(self.retriever, "top_k", self.settings.retrieval_top_k)),
                )
                node.documents = rank_documents(
                    contextual_question,
                    [*node.documents, *bridge_docs],
                    merge_limit,
                )
                if "vector_entity_bridge" not in node.retrieval_backends:
                    node.retrieval_backends.append("vector_entity_bridge")
                _cb(
                    event="node_bridge",
                    node_id=node.node_id,
                    query=bridge_query,
                    added=len(bridge_docs),
                )
            node.answer = self.generator.generate_for_node(
                contextual_question, node.documents, prior_hops=prior_hops
            )
            node.validation = self._validate_answer(contextual_question, node.answer, node.documents)
            node.source_consensus = node.validation.consensus_score

            if (
                node.validation.source_conflict
                and not conflict_attempted
                and self.conflict_auditor is not None
            ):
                conflict_attempted = True
                resolution = self.conflict_auditor.resolve(
                    contextual_question, node.answer, node.documents
                )
                if resolution is not None:
                    extra_docs = self.retriever.retrieve(resolution.query)
                    if extra_docs:
                        merge_limit = max(
                            4,
                            self.settings.retrieval_top_k + 1,
                            int(getattr(self.retriever, "top_k", self.settings.retrieval_top_k)),
                        )
                        node.documents = rank_documents(
                            contextual_question,
                            [*node.documents, *extra_docs],
                            merge_limit,
                        )
                        if "conflict_audit" not in node.retrieval_backends:
                            node.retrieval_backends.append("conflict_audit")
                        _cb(
                            event="node_conflict_audit",
                            node_id=node.node_id,
                            query=resolution.query,
                            reason=resolution.reason,
                            added=len(extra_docs),
                        )
                        node.answer = self.generator.generate_for_node(
                            contextual_question, node.documents, prior_hops=prior_hops
                        )
                        node.validation = self._validate_answer(contextual_question, node.answer, node.documents)
                        node.source_consensus = node.validation.consensus_score
            
            if node.validation.passed:
                node.status = "verified"
                if verified_cache is not None:
                    verified_cache[self._cache_key(base_question, prior_hops)] = node.answer
                break

            rationale = (node.validation.rationale if node.validation else "").lower()
            low_signal_failure = (
                (node.validation is not None and node.validation.confidence <= 0.15)
                and (
                    "insufficient evidence" in rationale
                    or "no evidence" in rationale
                    or "validation error" in rationale
                    or "judge failure" in rationale
                )
            )
            if low_signal_failure and attempt >= 1:
                skip_restructure = True
                break
            
            # Optimization: If category mismatch is the reason for failure, 
            # don't bother retrying the same question - jump to restructuring.
            if "[category mismatch]" in rationale:
                break
                
            refined = self.corrector.refine(retrieval_question, node.attempts)
            retrieval_question = self._safe_refine_query(retrieval_question, refined)

        # ART-R Restructuring Loop
        can_restructure = (
            not skip_restructure
            and node.validation is not None
            and not node.validation.passed
            and restructure_depth < restructure_cap
            and self.restructurer is not None
        )
        if can_restructure and deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= max(5.0, self.settings.llm_timeout_seconds + 2):
                can_restructure = False
                if node.validation is not None:
                    node.validation.rationale = (
                        f"{node.validation.rationale} (Skipped restructure: low time budget.)"
                    )

        if can_restructure:
            _cb(event="node_restructure", node_id=node.node_id, question=base_question)
            new_sub_nodes = self.restructurer.restructure(node, node.validation.rationale)
            if new_sub_nodes:
                node.children = new_sub_nodes
                node.was_restructured = True
                node.original_question = base_question
                # Now resolve this node as a group node
                self._resolve_tree(
                    node, 
                    prior_hops=prior_hops, 
                    progress_callback=_cb, 
                    verified_cache=verified_cache,
                    restructure_depth=restructure_depth + 1,
                    deadline=deadline,
                    max_retries=retry_cap,
                    max_restructures=restructure_cap,
                )
                return

        if not node.validation or not node.validation.passed:
            node.status = "needs_review"

        _cb(event="node_done",
            node_id=node.node_id,
            question=base_question,
            status=node.status,
            answer=self._strip_sources(node.answer),
            attempts=node.attempts,
            route=node.retrieval_route,
            backends=node.retrieval_backends,
            latency_ms=round(node.retrieval_latency_ms, 2))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _bridge_director_country_evidence(
        self, question: str, documents: list[RetrievedDocument]
    ) -> tuple[str | None, list[RetrievedDocument]]:
        """CRAG-style corrective action for director-country sub-questions.

        If a question asks for a director's country/nationality and evidence includes
        "directed by <Name>", run a focused vector query for that person. This
        reduces false `needs_review` caused by retrieving only film pages.
        """
        if not self._is_director_country_question(question):
            return None, []
        director_names = self._extract_director_names(documents)
        if not director_names:
            return None, []
        vector_backend = getattr(self.retriever, "vector_backend", None)
        if vector_backend is None or not hasattr(vector_backend, "search"):
            return None, []

        limit = max(
            2,
            min(6, int(getattr(self.retriever, "top_k", self.settings.retrieval_top_k)) + 1),
        )
        wants_nationality = "nationality" in question.lower()

        merged: list[RetrievedDocument] = []
        used_query: str | None = None
        for name in director_names[:2]:
            bridge_query = (
                f"What is the nationality of {name}?"
                if wants_nationality
                else f"What country is {name} from?"
            )
            used_query = bridge_query
            try:
                merged.extend(vector_backend.search(bridge_query, limit))
            except Exception:
                continue
        return used_query, merged

    def _validate_answer(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> ValidationResult:
        try:
            return self.validator.validate(answer, documents, question=question)
        except TypeError:
            return self.validator.validate(answer, documents)

    @staticmethod
    def _is_director_country_question(question: str) -> bool:
        lowered = question.lower()
        return "director" in lowered and (
            "country" in lowered or "nationality" in lowered
        )

    @staticmethod
    def _extract_director_names(documents: list[RetrievedDocument]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        patterns = [
            r"directed by ([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})",
            r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})\s*\([^)]*\)\s+was\s+a\s+[A-Za-z-]+\s+film director",
        ]
        for document in documents:
            text = document.content
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    candidate = re.sub(r"\s+", " ", match).strip(" .,:;")
                    if len(candidate.split()) < 2:
                        continue
                    lowered = candidate.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    names.append(candidate)
        return names

    def _safe_refine_query(self, previous: str, candidate: str) -> str:
        """Keep retries from drifting into generic 'RAG internals' phrasing."""
        candidate = (candidate or "").strip()
        if not candidate:
            return previous
        prev_entities = self._extract_query_entities(previous)
        if not prev_entities:
            return candidate
        cand_lower = candidate.lower()
        if any(entity in cand_lower for entity in prev_entities):
            return candidate
        return previous

    @staticmethod
    def _extract_query_entities(question: str) -> set[str]:
        matches = re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", question)
        stop = {
            "What",
            "Who",
            "Where",
            "When",
            "Which",
            "How",
            "Do",
            "Does",
            "Did",
            "Is",
            "Are",
            "Was",
            "Were",
            "The",
            "And",
            "Or",
        }
        return {match.lower() for match in matches if match not in stop}

    @staticmethod
    def _question_uses_anaphora(question: str) -> bool:
        lowered = question.lower()
        if lowered.startswith(("how ", "why ", "then ", "after ")):
            return True
        # Angle-bracket placeholders emitted by the LLM decomposer
        # e.g. "When was <director> born?" — always a dependent hop
        if re.search(r"<[^>]+>", question):
            return True
        markers = (
            " it ",
            " its ",
            " they ",
            " their ",
            " that ",
            " those ",
            " this ",
            " these ",
            " former ",
            " latter ",
            " previous ",
            " above ",
            " same as ",
            " that director",
            " the director from",
            " that person",
            " that team",
            " that film",
            " that movie",
        )
        padded = f" {lowered} "
        return any(marker in padded for marker in markers)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_query(question: str, prior_hops: list[tuple[str, str]]) -> str:
        """Append short prior-hop answers to the retrieval query so that rare
        named entities found in hop N-1 boost retrieval scores in hop N."""
        if not prior_hops:
            return question
        extra = " ".join(ans for _, ans in prior_hops)
        return f"{question} {extra}"

    @classmethod
    def _contextualize_question(
        cls,
        question: str,
        prior_hops: list[tuple[str, str]],
    ) -> str:
        if not prior_hops:
            return question

        entity = cls._extract_context_entity(prior_hops[-1][1])

        # --- Resolve angle-bracket placeholders like <director>, <team>, <person> ---
        # Do this regardless of whether entity was found via regex — use any
        # prominent proper noun extracted from the prior-hop answer.
        placeholder_match = re.search(r"<([^>]+)>", question)
        if placeholder_match:
            if entity:
                # Replace <whatever> with the extracted entity
                question = re.sub(r"<[^>]+>", entity, question)
            else:
                # Fall back: extract the longest capitalised noun phrase from the answer
                fallback = cls._extract_any_entity(prior_hops[-1][1])
                if fallback:
                    question = re.sub(r"<[^>]+>", fallback, question)
            return question

        if not entity:
            return question

        contextualized = re.sub(
            r"\b(?:this|that)\s+body\s+of\s+water\b",
            entity,
            question,
            flags=re.IGNORECASE,
        )
        contextualized = re.sub(
            r"\b(?:this|that)\s+(?:sea|gulf|bay|strait|lake|river|reservoir)\b",
            entity,
            contextualized,
            flags=re.IGNORECASE,
        )
        contextualized = re.sub(
            r"\bthat\s+baseball\s+team\b",
            entity,
            contextualized,
            flags=re.IGNORECASE,
        )
        return contextualized

    @staticmethod
    def _extract_context_entity(answer: str) -> str:
        cleaned = TreeQAPipeline._strip_sources(answer)

        # Water bodies
        water = re.search(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
            r"(?:Sea|Ocean|Bay|Strait|Reservoir|Lake|River))\b",
            cleaned,
        )
        if water:
            return water.group(1)

        gulf = re.search(r"\b(Gulf\s+of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b", cleaned)
        if gulf:
            return gulf.group(1)

        # Sports teams: "X won"
        team = re.search(
            r"\b(?:The\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s+won\b",
            cleaned,
        )
        if team:
            return team.group(1)

        # Person / director patterns: "directed by X", "director is X", "director was X"
        director = re.search(
            r"(?:directed by|director(?:\s+(?:is|was|of the film)?)?[:\s]+)"
            r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3})",
            cleaned,
        )
        if director:
            name = director.group(1).strip(" .,;")
            # Must have at least 2 parts to be a real full name, otherwise keep partial
            return name

        return ""

    @staticmethod
    def _extract_any_entity(answer: str) -> str:
        """Last-resort: return the longest capitalised multi-word noun phrase
        from the answer, excluding common sentence starters."""
        cleaned = TreeQAPipeline._strip_sources(answer)
        stop = {"The", "A", "An", "This", "That", "These", "Those",
                "He", "She", "It", "They", "His", "Her", "Its"}
        # Find all capitalised tokens sequences (proper noun phrases)
        phrases = re.findall(
            r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+)*)",
            cleaned,
        )
        candidates = [
            p.strip(" .,;")
            for p in phrases
            if p.split()[0] not in stop and len(p.split()) >= 1
        ]
        if not candidates:
            return ""
        # Prefer longer (more specific) phrases
        return max(candidates, key=len)

    @staticmethod
    def _strip_sources(text: str) -> str:
        return re.sub(r"\s*Sources:\s.*$", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _mark_timeout(node: QueryNode) -> None:
        if not node.answer:
            node.answer = f"Insufficient time budget to answer: {node.question}"
        node.status = "needs_review"
        node.validation = ValidationResult(
            passed=False,
            confidence=0.0,
            rationale="Time budget exceeded.",
        )

    @staticmethod
    def _hop_context(prior_hops: list[tuple[str, str]]) -> str:
        if not prior_hops:
            return ""
        return " | ".join(f"{q} => {a}" for q, a in prior_hops)

    @classmethod
    def _cache_key(cls, question: str, prior_hops: list[tuple[str, str]]) -> tuple[str, str]:
        return (question, cls._hop_context(prior_hops))

    def _flatten_leaves(self, root: QueryNode) -> list[QueryNode]:
        if root.is_leaf:
            return [root]
        leaves: list[QueryNode] = []
        for child in root.children:
            leaves.extend(self._flatten_leaves(child))
        return leaves

    def _build_group_validation(
        self, status: str, confidences: list[float]
    ) -> ValidationResult:
        average_confidence = sum(confidences) / len(confidences)
        rationale = (
            "All child nodes were verified."
            if status == "verified"
            else "At least one child node requires review."
        )
        return ValidationResult(
            passed=status == "verified",
            confidence=average_confidence,
            rationale=rationale,
        )
