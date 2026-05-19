from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import logging
import re
import time

from hamhrag.backends.graph import GraphBackend, MemoryGraphBackend
from hamhrag.backends.vector import MemoryVectorBackend, VectorBackend
from hamhrag.models import RetrievedDocument
from hamhrag.retrieval.scoring import rank_documents


@dataclass(slots=True)
class RoutePlan:
    route: str
    reason: str
    query_type: str
    vector_limit: int
    graph_limit: int
    preferred_backend: str
    allow_fallback: bool
    confidence_threshold: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalTrace:
    route: str
    query_type: str
    reason: str
    signals: dict[str, float]
    vector_limit: int
    graph_limit: int
    backends_used: list[str] = field(default_factory=list)
    fallback_used: bool = False
    vector_latency_ms: float | None = None
    graph_latency_ms: float | None = None
    total_latency_ms: float = 0.0
    documents_found: int = 0


class QueryRouter:
    """Heuristic router that picks the fastest reliable retrieval path."""

    _RELATION_TERMS = {
        "who", "where", "when", "whose", "director", "founded", "founded by",
        "born", "capital", "parent", "owner", "ceo", "relationship", "between",
        "located", "country", "nationality", "married", "invented", "authored",
    }
    _MULTI_HOP_TERMS = {
        "compare", "difference", "between", "both", "versus", "vs", "after",
        "before", "then", "first", "second", "which", "more",
    }
    _EXPLANATION_TERMS = {
        "explain", "describe", "summarize", "overview", "impact", "benefits",
        "workflow", "process", "steps", "method", "architecture", "why", "how",
    }

    def __init__(self, base_top_k: int, min_top_k: int = 2, max_top_k: int = 24) -> None:
        self.base_top_k = max(1, base_top_k)
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

    def route(self, question: str) -> RoutePlan:
        lowered = question.lower()
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered)
        token_set = set(tokens)

        relation_hits = self._count_keyword_hits(lowered, token_set, self._RELATION_TERMS)
        explanation_hits = self._count_keyword_hits(lowered, token_set, self._EXPLANATION_TERMS)
        multi_hop_hits = self._count_keyword_hits(lowered, token_set, self._MULTI_HOP_TERMS)

        has_comparison_pattern = bool(
            re.search(r"\b(compare|difference|between|vs\.?|versus)\b", lowered)
        )
        has_chained_pattern = bool(
            re.search(r"\b(and then|first .* then|before .* after|after .* before)\b", lowered)
        )
        multi_hop = bool(multi_hop_hits or has_comparison_pattern or has_chained_pattern)
        entity_hint = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", question))
        is_short = len(tokens) <= 8

        graph_score = (
            relation_hits * 0.3
            + (0.35 if multi_hop else 0.0)
            + (0.12 if entity_hint else 0.0)
            + (0.08 if question.strip().lower().startswith(("who", "where", "when")) else 0.0)
        )
        vector_score = (
            explanation_hits * 0.28
            + (0.18 if not multi_hop else 0.0)
            + (0.08 if len(tokens) > 12 else 0.0)
            + (0.08 if question.strip().lower().startswith(("explain", "describe", "summarize")) else 0.0)
        )

        complexity = min(
            1.0,
            (len(tokens) / 16.0)
            + (0.28 if multi_hop else 0.0)
            + (0.16 if relation_hits >= 2 else 0.0),
        )
        dynamic_top_k = int(round(self.base_top_k * (0.8 + (complexity * 0.9))))
        dynamic_top_k = max(self.min_top_k, min(dynamic_top_k, self.max_top_k))
        if re.search(r"\bteam\b.*\bwon\b.*\bworld series\b|\bworld series\b.*\bwon\b", lowered):
            dynamic_top_k = min(self.max_top_k, max(dynamic_top_k, 10))
        if entity_hint and re.search(
            r"\b(body of water|sea|ocean|bay|strait|lake|river|reservoir|located)\b",
            lowered,
        ):
            dynamic_top_k = min(self.max_top_k, max(dynamic_top_k, 20))

        signals = {
            "graph_score": round(graph_score, 3),
            "vector_score": round(vector_score, 3),
            "relation_hits": float(relation_hits),
            "explanation_hits": float(explanation_hits),
            "multi_hop_hits": float(multi_hop_hits),
            "complexity": round(complexity, 3),
            "query_tokens": float(len(tokens)),
        }

        if multi_hop or has_comparison_pattern:
            return RoutePlan(
                route="hybrid_parallel",
                reason="Detected multi-hop/comparison cues, so both backends run in parallel for coverage.",
                query_type="multi_hop",
                vector_limit=dynamic_top_k,
                graph_limit=dynamic_top_k,
                preferred_backend="parallel",
                allow_fallback=True,
                confidence_threshold=0.72,
                signals=signals,
            )

        if (graph_score - vector_score) >= 0.35:
            return RoutePlan(
                route="graph_first",
                reason="Strong entity/relationship cues; graph retrieval with vector fallback.",
                query_type="entity_relation",
                vector_limit=max(self.min_top_k, dynamic_top_k - 1),
                graph_limit=dynamic_top_k + 1,
                preferred_backend="graph",
                allow_fallback=True,
                confidence_threshold=0.67,
                signals=signals,
            )

        if (vector_score - graph_score) >= 0.35 and is_short:
            return RoutePlan(
                route="vector_only",
                reason="Simple descriptive query detected; vector-first route minimizes latency with graph fallback on miss.",
                query_type="single_hop_descriptive",
                vector_limit=dynamic_top_k,
                graph_limit=max(self.min_top_k, dynamic_top_k - 1),
                preferred_backend="vector",
                allow_fallback=True,
                confidence_threshold=0.0,
                signals=signals,
            )

        if vector_score > graph_score:
            return RoutePlan(
                route="vector_first",
                reason="Mostly descriptive query, so vector runs first with graph fallback when confidence is weak.",
                query_type="mixed_descriptive",
                vector_limit=dynamic_top_k,
                graph_limit=max(self.min_top_k, dynamic_top_k - 1),
                preferred_backend="vector",
                allow_fallback=True,
                confidence_threshold=0.69,
                signals=signals,
            )

        return RoutePlan(
            route="hybrid_parallel",
            reason="Ambiguous intent: running both backends in parallel for robust recall.",
            query_type="mixed_ambiguous",
            vector_limit=dynamic_top_k,
            graph_limit=dynamic_top_k,
            preferred_backend="parallel",
            allow_fallback=True,
            confidence_threshold=0.72,
            signals=signals,
        )

    @staticmethod
    def _count_keyword_hits(lowered: str, tokens: set[str], keywords: set[str]) -> int:
        hits = 0
        for kw in keywords:
            if " " in kw:
                if kw in lowered:
                    hits += 1
            elif kw in tokens:
                hits += 1
        return hits


class HybridRetriever:
    """Combines vector and graph retrieval through swappable backends."""

    def __init__(
        self,
        vector_backend: VectorBackend | None = None,
        graph_backend: GraphBackend | None = None,
        top_k: int = 3,
    ) -> None:
        self.vector_backend = vector_backend or MemoryVectorBackend()
        self.graph_backend = graph_backend or MemoryGraphBackend()
        self.top_k = top_k
        self.router = QueryRouter(base_top_k=top_k)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._log = logging.getLogger(__name__)

    def retrieve(self, question: str) -> list[RetrievedDocument]:
        documents, _trace = self.retrieve_with_trace(question)
        return documents

    def retrieve_with_trace(self, question: str) -> tuple[list[RetrievedDocument], RetrievalTrace]:
        plan = self.router.route(question)
        trace = RetrievalTrace(
            route=plan.route,
            query_type=plan.query_type,
            reason=plan.reason,
            signals=dict(plan.signals),
            vector_limit=plan.vector_limit,
            graph_limit=plan.graph_limit,
        )

        started_at = time.perf_counter()
        documents: list[RetrievedDocument] = []

        if plan.route == "vector_only":
            vector_docs = self._run_vector(question, plan.vector_limit, trace)
            documents.extend(vector_docs)
            if not vector_docs and plan.allow_fallback and plan.graph_limit > 0:
                trace.fallback_used = True
                documents.extend(self._run_graph(question, plan.graph_limit, trace))
        elif plan.route == "graph_first":
            # If fallback is likely needed, use parallel execution to avoid sequential latency
            will_need_fallback = self._predict_graph_fallback_needed(question, plan)
            if will_need_fallback and plan.allow_fallback:
                documents.extend(self._run_hybrid_parallel(question, plan, trace))
            else:
                graph_docs = self._run_graph(question, plan.graph_limit, trace)
                documents.extend(graph_docs)
                if self._needs_fallback(
                    graph_docs, plan.confidence_threshold, plan.allow_fallback
                ) or self._graph_docs_need_vector_fallback(question, graph_docs):
                    trace.fallback_used = True
                    documents.extend(self._run_vector(question, plan.vector_limit, trace))
        elif plan.route == "vector_first":
            vector_docs = self._run_vector(question, plan.vector_limit, trace)
            documents.extend(vector_docs)
            if self._needs_fallback(vector_docs, plan.confidence_threshold, plan.allow_fallback):
                trace.fallback_used = True
                documents.extend(self._run_graph(question, plan.graph_limit, trace))
        else:
            documents.extend(self._run_hybrid_parallel(question, plan, trace))

        cleaned = self._prune_off_topic_graph_docs(question, documents)
        ranked = rank_documents(question, cleaned, self.top_k) if cleaned else []
        trace.total_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        trace.documents_found = len(ranked)
        return ranked, trace

    def _predict_graph_fallback_needed(self, question: str, plan: RoutePlan) -> bool:
        """Predict if graph_first will likely need vector fallback without running graph.
        
        Returns True if:
        - Question has entity tokens but few tokens overall (suggests pure lookup)
        - Query is too abstract for graph (no relation terms, lots of explanation terms)
        - This allows us to use parallel execution preemptively instead of sequential fallback
        """
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", question.lower())
        if len(tokens) <= 4:
            return False  # Too few tokens, likely simple lookup (keep sequential)
        
        entity_tokens = self._extract_entity_tokens(question)
        if len(entity_tokens) < 2:
            return False  # No clear entities, sequential is fine
        
        lowered = question.lower()
        relation_hits = self.router._count_keyword_hits(lowered, set(tokens), QueryRouter._RELATION_TERMS)
        explanation_hits = self.router._count_keyword_hits(lowered, set(tokens), QueryRouter._EXPLANATION_TERMS)
        
        # If question is mostly explanation-oriented despite entity hint, fallback likely needed
        return explanation_hits > 0 and relation_hits <= 1

    @staticmethod
    def _count_keyword_hits(lowered: str, tokens: set[str], keywords: set[str]) -> int:
        hits = 0
        for kw in keywords:
            if " " in kw:
                if kw in lowered:
                    hits += 1
            elif kw in tokens:
                hits += 1
        return hits

    @staticmethod
    def _needs_fallback(
        docs: list[RetrievedDocument], threshold: float, allow_fallback: bool
    ) -> bool:
        if not allow_fallback:
            return False
        if not docs:
            return True
        return docs[0].score < threshold

    def _run_vector(
        self, question: str, limit: int, trace: RetrievalTrace
    ) -> list[RetrievedDocument]:
        if limit <= 0:
            return []
        started_at = time.perf_counter()
        try:
            docs = self.vector_backend.search(question, limit)
            trace.backends_used.append("vector")
            return docs
        except Exception as exc:
            self._log.warning("Vector search failed: %s", exc)
            return []
        finally:
            trace.vector_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

    def _run_graph(
        self, question: str, limit: int, trace: RetrievalTrace
    ) -> list[RetrievedDocument]:
        if limit <= 0:
            return []
        started_at = time.perf_counter()
        try:
            docs = self.graph_backend.search(question, limit)
            trace.backends_used.append("graph")
            return docs
        except Exception as exc:
            self._log.warning("Graph search failed: %s", exc)
            return []
        finally:
            trace.graph_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

    def _run_hybrid_parallel(
        self, question: str, plan: RoutePlan, trace: RetrievalTrace
    ) -> list[RetrievedDocument]:
        futures: dict[str, concurrent.futures.Future[list[RetrievedDocument]]] = {}
        submit_times: dict[str, float] = {}
        if plan.vector_limit > 0:
            submit_times["vector"] = time.perf_counter()
            futures["vector"] = self._executor.submit(self.vector_backend.search, question, plan.vector_limit)
        if plan.graph_limit > 0:
            submit_times["graph"] = time.perf_counter()
            futures["graph"] = self._executor.submit(self.graph_backend.search, question, plan.graph_limit)

        merged: list[RetrievedDocument] = []
        future_to_backend = {future: backend for backend, future in futures.items()}
        for future in concurrent.futures.as_completed(future_to_backend):
            backend = future_to_backend[future]
            try:
                docs = future.result()
                merged.extend(docs)
                trace.backends_used.append(backend)
            except Exception as exc:
                self._log.warning("%s search failed: %s", backend.capitalize(), exc)
            finally:
                started_at = submit_times.get(backend, time.perf_counter())
                elapsed = round((time.perf_counter() - started_at) * 1000.0, 2)
                if backend == "vector":
                    trace.vector_latency_ms = elapsed
                elif backend == "graph":
                    trace.graph_latency_ms = elapsed
        return merged

    def _prune_off_topic_graph_docs(
        self, question: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Drop graph distractors when entity-rich queries already have vector evidence.

        Local graph indices may include generic framework facts (e.g. retrieval internals)
        that match refinement boilerplate but not the user entity. For named-entity
        questions, keep only graph docs that mention at least one query entity token
        when vector evidence is available.
        """
        if not documents:
            return documents
        has_vector = any(doc.source_type == "vector" for doc in documents)
        if not has_vector:
            return documents

        entity_tokens = self._extract_entity_tokens(question)
        if len(entity_tokens) < 2:
            return documents

        filtered: list[RetrievedDocument] = []
        for doc in documents:
            if doc.source_type != "graph":
                filtered.append(doc)
                continue
            content = doc.content.lower()
            if any(token in content for token in entity_tokens):
                filtered.append(doc)
        return filtered or documents

    def _graph_docs_need_vector_fallback(
        self, question: str, graph_docs: list[RetrievedDocument]
    ) -> bool:
        if not graph_docs:
            return False
        entity_tokens = self._extract_entity_tokens(question)
        if len(entity_tokens) < 2:
            return False
        for doc in graph_docs:
            content = doc.content.lower()
            if any(token in content for token in entity_tokens):
                return False
        return True

    @staticmethod
    def _extract_entity_tokens(question: str) -> set[str]:
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9'-]{2,}\b", question)
        if not tokens:
            return set()
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
            "Can",
            "Could",
            "Would",
            "Should",
            "The",
            "And",
            "Or",
        }
        return {token.lower() for token in tokens if token not in stop}
