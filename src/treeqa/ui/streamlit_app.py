from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from treeqa.pipeline import TreeQAPipeline


def main() -> None:
    st.set_page_config(page_title="TreeQA", layout="wide")
    st.title("Project TreeQA")
    st.caption("Hallucination-aware multi-hop RAG scaffold")

    default_query = (
        "Explain how a logic tree, hybrid retrieval, and validation loop reduce hallucinations."
    )
    query = st.text_area("Question", value=default_query, height=120)

    if st.button("Run workflow", type="primary"):
        pipeline = TreeQAPipeline()
        result = pipeline.run(query)
        st.subheader("Final answer")
        st.write(result.final_answer)
        st.subheader("Logic tree")
        st.json(asdict(result))


if __name__ == "__main__":
    main()

