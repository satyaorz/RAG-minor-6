from __future__ import annotations

import re

from treeqa.models import RetrievedDocument


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
        if token not in STOPWORDS
    ]


def lexical_score(question: str, content: str) -> float:
    question_tokens = tokenize(question)
    if not question_tokens:
        return 0.0
    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0

    question_set = set(question_tokens)
    content_set = set(content_tokens)
    overlap = len(question_set & content_set)
    ordered_phrase = " ".join(question_tokens[: min(3, len(question_tokens))])
    phrase_bonus = 0.15 if ordered_phrase and ordered_phrase in content.lower() else 0.0
    density_bonus = min(0.2, overlap / max(len(content_tokens), 1))
    return (overlap / len(question_set)) + phrase_bonus + density_bonus


def select_relevant_snippet(content: str, question: str, max_sentences: int = 2) -> str:
    normalized = normalize_text(content)
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    if len(sentences) <= max_sentences:
        return normalized
    ranked = sorted(
        sentences,
        key=lambda sentence: lexical_score(question, sentence),
        reverse=True,
    )
    chosen = [sentence for sentence in ranked[:max_sentences] if sentence.strip()]
    if not chosen:
        return normalized
    ordered = [sentence for sentence in sentences if sentence in chosen]
    return " ".join(ordered)


def rank_documents(question: str, documents: list[RetrievedDocument], limit: int) -> list[RetrievedDocument]:
    ranked: list[RetrievedDocument] = []
    seen_content: set[str] = set()
    source_type_counts: dict[str, int] = {}
    for document in sorted(
        documents,
        key=lambda item: (
            lexical_score(question, item.content)
            + _entity_overlap_bonus(question, item)
            + _answer_type_bonus(question, item.content)
            + item.score
        ),
        reverse=True,
    ):
        normalized = normalize_text(document.content)
        if normalized in seen_content:
            continue
        source_penalty = source_type_counts.get(document.source_type, 0) * 0.03
        adjusted_score = (
            lexical_score(question, normalized)
            + _entity_overlap_bonus(question, document)
            + _answer_type_bonus(question, normalized)
            + document.score
            - source_penalty
        )
        ranked.append(
            RetrievedDocument(
                source_id=document.source_id,
                source_type=document.source_type,
                content=normalized,
                score=adjusted_score,
            )
        )
        seen_content.add(normalized)
        source_type_counts[document.source_type] = source_type_counts.get(document.source_type, 0) + 1
        if len(ranked) >= limit:
            break
    return ranked


def _entity_overlap_bonus(question: str, document: RetrievedDocument) -> float:
    entities = [
        entity.lower()
        for entity in re.findall(r"\b[A-Z][A-Za-z0-9'-]{2,}(?:\s+[A-Z][A-Za-z0-9'-]{2,})*\b", question)
        if entity not in {"What", "Who", "Where", "When", "Which", "How", "The"}
    ]
    if not entities:
        return 0.0
    haystack = f"{document.source_id} {document.content}".lower().replace("_", " ")
    hits = sum(1 for entity in entities if entity in haystack)
    return min(1.5, hits * 1.0)


def _answer_type_bonus(question: str, content: str) -> float:
    lowered = question.lower()
    if "body of water" not in lowered:
        return 0.0
    if re.search(
        r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
        r"(?:Sea|Ocean|Bay|Strait|Reservoir|Lake|River)\b",
        content,
    ):
        return 1.0
    return 0.0
