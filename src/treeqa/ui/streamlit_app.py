from __future__ import annotations

from dataclasses import asdict
from html import escape

import streamlit as st

from treeqa.models import QueryNode
from treeqa.pipeline import TreeQAPipeline


def main() -> None:
    st.set_page_config(page_title="TreeQA", layout="wide")
    st.title("Project TreeQA")
    st.caption("Hallucination-aware multi-hop RAG scaffold")

    default_query = (
        "Explain how a logic tree, hybrid retrieval, and validation loop reduce hallucinations."
    )
    query = st.text_area("Question", value=default_query, height=120)

    mode = st.radio(
        "Mode",
        ["⚡ Normal RAG", "🌳 TreeQA", "⚖ Compare Both"],
        horizontal=True,
        help=(
            "Normal RAG: single-hop retrieve + LLM answer. "
            "TreeQA: decompose → retrieve → validate → synthesise. "
            "Compare: run both side-by-side."
        ),
    )

    run_label = {"⚡ Normal RAG": "Run Normal RAG", "🌳 TreeQA": "Run TreeQA", "⚖ Compare Both": "Compare Both"}[mode]

    if st.button(run_label, type="primary"):
        pipeline = TreeQAPipeline()

        if mode == "⚡ Normal RAG":
            _render_rag(pipeline, query)

        elif mode == "🌳 TreeQA":
            result = pipeline.run(query)
            st.subheader("Final answer")
            st.write(result.final_answer)
            _render_summary(result.root)
            st.subheader("Logic tree")
            _render_node(result.root)
            st.subheader("Raw payload")
            st.json(asdict(result))

        else:  # Compare Both
            col_rag, col_tqa = st.columns(2)
            docs = pipeline.retriever.retrieve(query)
            rag_answer = pipeline.generator.generate_for_node(query, docs)

            with col_rag:
                st.subheader("⚡ Normal RAG")
                st.info(rag_answer)
                top_score = docs[0].score if docs else 0.0
                st.caption(f"Retrieval confidence: {top_score:.2%} · {len(docs)} doc(s) · no decomposition")
                with st.expander("Retrieved documents"):
                    for doc in docs:
                        st.markdown(f"- `{doc.source_type}:{doc.source_id}` score `{doc.score:.3f}`")
                        st.write(doc.content[:400])

            with col_tqa:
                st.subheader("🌳 TreeQA")
                result = pipeline.run(query)
                st.success(result.final_answer)
                nodes = result.root.iter_nodes()
                verified = sum(1 for n in nodes if n.status == "verified")
                st.caption(f"Status: {result.root.status} · {verified} node(s) verified")
                _render_summary(result.root)

            st.subheader("Full TreeQA logic tree")
            _render_node(result.root)


def _render_rag(pipeline: TreeQAPipeline, query: str) -> None:
    """Render Normal RAG results: LLM answer + retrieved evidence."""
    docs = pipeline.retriever.retrieve(query)
    answer = pipeline.generator.generate_for_node(query, docs)

    st.subheader("⚡ Normal RAG Answer")
    st.write(answer)

    if docs:
        top_score = docs[0].score
        st.caption(
            f"Retrieval confidence: {top_score:.2%} · "
            f"{len(docs)} document(s) retrieved · "
            "no query decomposition or hallucination validation"
        )

    st.subheader(f"Retrieved Evidence ({len(docs)} document(s))")
    for doc in docs:
        with st.expander(
            f"{doc.source_type}:{doc.source_id}  —  score {doc.score:.3f}",
            expanded=False,
        ):
            st.write(doc.content)


def _render_summary(root: QueryNode) -> None:
    nodes = root.iter_nodes()
    verified = sum(1 for node in nodes if node.status == "verified")
    needs_review = sum(1 for node in nodes if node.status == "needs_review")
    total = len(nodes)
    column_one, column_two, column_three = st.columns(3)
    column_one.metric("Nodes", total)
    column_two.metric("Verified", verified)
    column_three.metric("Needs review", needs_review)


def _render_node(node: QueryNode, depth: int = 0) -> None:
    palette = _status_palette(node.status)
    confidence = (
        f"{node.validation.confidence:.2f}" if node.validation is not None else "n/a"
    )
    margin = depth * 24
    st.markdown(
        f"""
        <div style="
            margin-left: {margin}px;
            padding: 14px 16px;
            border-left: 6px solid {palette["border"]};
            background: {palette["background"]};
            border-radius: 10px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 0.8rem; color: #475569; margin-bottom: 6px;">
                {escape(node.node_id)} · {escape(node.status)} · attempts {node.attempts} · confidence {escape(confidence)}
            </div>
            <div style="font-weight: 600; margin-bottom: 8px;">{escape(node.question)}</div>
            <div>{escape(node.answer or "Pending answer")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if node.documents:
        with st.expander(f"Evidence for {node.node_id}", expanded=False):
            for document in node.documents:
                st.markdown(
                    f"- `{document.source_type}:{document.source_id}` score `{document.score:.2f}`"
                )
                st.write(document.content)
    for child in node.children:
        _render_node(child, depth + 1)


def _status_palette(status: str) -> dict[str, str]:
    if status == "verified":
        return {"background": "#ecfdf3", "border": "#16a34a"}
    if status == "needs_review":
        return {"background": "#fef3c7", "border": "#d97706"}
    return {"background": "#e2e8f0", "border": "#64748b"}


if __name__ == "__main__":
    main()
