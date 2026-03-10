"""
Benchmark dataset loader for TreeQA.

Streams rows from Hugging Face Hub, extracts the text corpus (documents),
writes deduplicated markdown files to data/documents/datasets/<key>/,
and saves a small Q&A sample to data/benchmark/ for immediate testing.

Difficulty guide
----------------
earsy    — single-hop: answer is in one passage (SQuAD, MS MARCO)
medium   — multi-hop comparison: "Which X was founded earlier, A or B?"
hard     — multi-hop bridge: you MUST find intermediate entity before you can
            retrieve the second document (2WikiMultiHopQA bridge, MuSiQue)

Why Normal RAG fails on "hard" questions
----------------------------------------
Normal RAG retrieves top-k docs for the *original* question.  When the
question hides the bridge entity ("Where was the director of Gravity born?"),
Retrieval can't find the birthplace doc because it never searched for Cuarón.
TreeQA decomposes before retrieving:  sub-q1 → "Who directed Gravity?" →
retrieve → answer Cuarón; sub-q2 → "Where was Alfonso Cuarón born?" → retrieve
→ answer Mexico City.  Normal RAG guesses or hallucinates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


MAX_ROWS_HARD_CAP = 2_000

SUPPORTED_DATASETS: dict[str, dict] = {
    "hotpot_qa": {
        "name": "HotpotQA",
        "description": "Multi-hop QA: each question requires reasoning across 2+ Wikipedia articles. Good starting point.",
        "hf_name": "hotpot_qa",
        "hf_config": "distractor",
        "splits": ["train", "validation"],
        "type": "multi_hop",
        "difficulty": "medium",
        "difficulty_note": "LLM often solves these via context window even without decomposition.",
        "example_question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "default_max_rows": 300,
    },
    "2wikimultihopqa": {
        "name": "2WikiMultiHopQA",
        "description": (
            "Hard bridge questions: the intermediate entity is HIDDEN from the question. "
            "Normal RAG cannot retrieve the second document without first answering the first hop. "
            "This is where TreeQA shines."
        ),
        "hf_name": "framolfese/2WikiMultihopQA",
        "hf_config": "default",
        "splits": ["train", "validation", "test"],
        "type": "multi_hop",
        "difficulty": "hard",
        "difficulty_note": (
            'Bridge type: e.g. \"Where was the screenwriter of [Film X] born?\" '
            "\u2014 RAG can't find the birthplace without first knowing who wrote the film."
        ),
        "example_question": "Where was the director of film Blank Check born?",
        "default_max_rows": 200,
    },
    "musique": {
        "name": "MuSiQue",
        "description": (
            "Multi-step questions requiring 2\u20134 reasoning hops. "
            "Specifically constructed so that single-hop shortcuts fail. "
            "Hardest benchmark for Normal RAG."
        ),
        "hf_name": "dgslibisey/MuSiQue",
        "hf_config": "default",
        "splits": ["train", "validation"],
        "type": "multi_hop",
        "difficulty": "hard",
        "difficulty_note": "2\u20134 hops required. Single-hop retrieval gets the wrong answer >80% of the time.",
        "example_question": "The arena where the Lewiston Maineiacs played their home games was opened in what year?",
        "default_max_rows": 200,
    },
    "trivia_qa": {
        "name": "TriviaQA",
        "description": "Trivia questions paired with Wikipedia & web evidence passages.",
        "hf_name": "trivia_qa",
        "hf_config": "rc",
        "splits": ["train", "validation"],
        "type": "single_hop",
        "difficulty": "easy",
        "difficulty_note": "Answer is in a single passage. Normal RAG handles these well.",
        "example_question": "The Nicobar Scops Owl is native to what group of islands?",
        "default_max_rows": 300,
    },
    "squad_v2": {
        "name": "SQuAD 2.0",
        "description": "Reading-comprehension passages with answerable & unanswerable questions.",
        "hf_name": "squad_v2",
        "hf_config": None,
        "splits": ["train", "validation"],
        "type": "single_hop",
        "difficulty": "easy",
        "difficulty_note": "Answer is directly in the passage. Normal RAG \u2248 TreeQA here.",
        "example_question": "In what country is Normandy located?",
        "default_max_rows": 500,
    },
    "ms_marco": {
        "name": "MS MARCO",
        "description": "Web search passages from real Bing queries (Microsoft Reading Comprehension).",
        "hf_name": "ms_marco",
        "hf_config": "v1.1",
        "splits": ["train", "validation"],
        "type": "single_hop",
        "difficulty": "easy",
        "difficulty_note": "Single web passage per question. Good for baseline comparison.",
        "example_question": "what is a primary source?",
        "default_max_rows": 300,
    },
}


@dataclass
class DatasetLoadResult:
    dataset_key: str
    dataset_name: str
    split: str
    rows_processed: int
    docs_written: int
    output_dir: str
    sample_questions: list[dict]


def load_and_write_corpus(
    dataset_key: str,
    split: str,
    max_rows: int,
    data_dir: Path,
) -> DatasetLoadResult:
    """Stream `max_rows` rows from HuggingFace Hub, extract text corpus,
    write deduplicated markdown files, and return stats + sample Q&A pairs.

    The `datasets` package must be installed::

        pip install datasets
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for benchmark loading. "
            "Install with:  pip install datasets"
        ) from exc

    cfg = SUPPORTED_DATASETS.get(dataset_key)
    if not cfg:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Supported: {list(SUPPORTED_DATASETS)}"
        )

    max_rows = min(max(1, max_rows), MAX_ROWS_HARD_CAP)

    load_kwargs: dict = {"split": split, "streaming": True, "trust_remote_code": False}
    if cfg["hf_config"]:
        ds = hf_load(cfg["hf_name"], cfg["hf_config"], **load_kwargs)
    else:
        ds = hf_load(cfg["hf_name"], **load_kwargs)

    out_dir = data_dir / "documents" / "datasets" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = _EXTRACTORS[dataset_key]
    doc_map: dict[str, list[str]] = {}   # title → unique paragraphs (insertion-ordered)
    sample_questions: list[dict] = []
    rows_processed = 0

    for row in ds:
        if rows_processed >= max_rows:
            break
        try:
            extracted = extractor(row)
        except Exception:
            rows_processed += 1
            continue

        for title, paragraphs in extracted["docs"]:
            bucket = doc_map.setdefault(title, [])
            seen: set[str] = set(bucket)
            for p in paragraphs:
                p = p.strip()
                if p and p not in seen:
                    bucket.append(p)
                    seen.add(p)

        q = extracted.get("question", "")
        a = extracted.get("answer", "")
        qtype = extracted.get("question_type", "")
        if q and len(sample_questions) < 20:
            sample_questions.append({"question": q, "answer": a, "question_type": qtype})

        rows_processed += 1

    # Write deduplicated markdown files
    docs_written = 0
    for title, paragraphs in doc_map.items():
        if not paragraphs:
            continue
        safe = _safe_filename(title)
        path = out_dir / f"{safe}.md"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {title}\n\n")
            for p in paragraphs:
                fh.write(p.rstrip() + "\n\n")
        docs_written += 1

    # Save sample benchmark Q&A pairs
    if sample_questions:
        bench_dir = data_dir / "benchmark"
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_path = bench_dir / f"{dataset_key}_{split}_sample.jsonl"
        with bench_path.open("w", encoding="utf-8") as fh:
            for qa in sample_questions:
                fh.write(json.dumps(qa, ensure_ascii=True) + "\n")

    return DatasetLoadResult(
        dataset_key=dataset_key,
        dataset_name=cfg["name"],
        split=split,
        rows_processed=rows_processed,
        docs_written=docs_written,
        output_dir=str(out_dir),
        sample_questions=sample_questions,
    )


# ── Per-dataset corpus extractors ─────────────────────────────────────────────

def _extract_hotpotqa(row: dict) -> dict:
    """context = {'title': [...], 'sentences': [[s, s, ...], ...]}"""
    ctx = row.get("context") or {}
    titles = ctx.get("title", []) if isinstance(ctx, dict) else []
    sentence_lists = ctx.get("sentences", []) if isinstance(ctx, dict) else []
    docs = []
    for title, sents in zip(titles, sentence_lists):
        if sents:
            docs.append((str(title), [" ".join(str(s) for s in sents)]))
    return {
        "docs": docs,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "question_type": row.get("type", ""),
    }


def _extract_2wikimultihop(row: dict) -> dict:
    """Same context shape as HotpotQA.  Type field distinguishes bridge / comparison / etc."""
    ctx = row.get("context") or {}
    # Two possible layouts depending on dataset version
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentence_lists = ctx.get("sentences", [])
        docs = [
            (str(t), [" ".join(str(s) for s in sents)])
            for t, sents in zip(titles, sentence_lists)
            if sents
        ]
    elif isinstance(ctx, list):
        # [[title, [sentences]], ...]
        docs = [
            (str(item[0]), [" ".join(str(s) for s in item[1])])
            for item in ctx
            if len(item) >= 2 and item[1]
        ]
    else:
        docs = []
    return {
        "docs": docs,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "question_type": row.get("type", ""),  # bridge | comparison | inference | compositional
    }


def _extract_musique(row: dict) -> dict:
    """paragraphs = [{'idx': int, 'title': str, 'paragraph_text': str, 'is_supporting': bool}]"""
    paragraphs = row.get("paragraphs") or []
    docs = [
        (str(p.get("title") or f"para-{p.get('idx', i)}"), [str(p.get("paragraph_text", ""))])
        for i, p in enumerate(paragraphs)
        if p.get("paragraph_text", "").strip()
    ]
    answer_aliases = row.get("answer_aliases") or []
    answer = row.get("answer") or (answer_aliases[0] if answer_aliases else "")
    # MuSiQue labels answerable vs unanswerable
    qtype = "answerable" if row.get("answerable", True) else "unanswerable"
    return {
        "docs": docs,
        "question": row.get("question", ""),
        "answer": str(answer),
        "question_type": qtype,
    }


def _extract_triviaqa(row: dict) -> dict:
    """entity_pages = {'title': [...], 'wiki_context': [...]}"""
    ep = row.get("entity_pages") or {}
    titles = ep.get("title", []) if isinstance(ep, dict) else []
    contexts = ep.get("wiki_context", []) if isinstance(ep, dict) else []
    docs = [(str(t), [str(c)]) for t, c in zip(titles, contexts) if c and str(c).strip()]
    # Fall back to search_results when entity_pages is empty
    if not docs:
        sr = row.get("search_results") or {}
        if isinstance(sr, dict):
            sr_titles = sr.get("title", [])
            sr_ctxs = sr.get("search_context", [])
            docs = [
                (str(t), [str(c)])
                for t, c in zip(sr_titles, sr_ctxs)
                if c and str(c).strip()
            ]
    answer_block = row.get("answer") or {}
    answer = (
        answer_block.get("value", "")
        if isinstance(answer_block, dict)
        else str(answer_block)
    )
    return {"docs": docs, "question": row.get("question", ""), "answer": answer}


def _extract_squad(row: dict) -> dict:
    """Each row is a single context passage + question."""
    title = str(row.get("title") or "passage")
    context = str(row.get("context") or "").strip()
    docs = [(title, [context])] if context else []
    answers = row.get("answers") or {}
    answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
    answer = answer_texts[0] if answer_texts else ""
    return {"docs": docs, "question": row.get("question", ""), "answer": answer}


def _extract_msmarco(row: dict) -> dict:
    """passages = {'passage_text': [...], 'is_selected': [...]}"""
    passages = row.get("passages") or {}
    texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
    qid = str(row.get("query_id") or "q")
    docs = [
        (f"{qid}-p{i}", [str(t)])
        for i, t in enumerate(texts)
        if t and str(t).strip()
    ]
    answers = row.get("answers") or []
    answer = answers[0] if isinstance(answers, list) and answers else ""
    return {"docs": docs, "question": str(row.get("query") or ""), "answer": answer}


_EXTRACTORS = {
    "hotpot_qa":       _extract_hotpotqa,
    "2wikimultihopqa": _extract_2wikimultihop,
    "musique":         _extract_musique,
    "trivia_qa":       _extract_triviaqa,
    "squad_v2":        _extract_squad,
    "ms_marco":        _extract_msmarco,
}


def _safe_filename(title: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"[\s-]+", "_", safe).strip("_")
    return (safe[:80] or "doc")
