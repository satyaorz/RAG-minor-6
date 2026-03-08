# TreeQA Overview

Project TreeQA is a hallucination-aware retrieval-augmented generation system for multi-hop question answering.
It decomposes complex questions into verifiable sub-questions, retrieves both unstructured and structured evidence,
validates each sub-answer, and retries when evidence is weak.

The system is designed to visualize a logic tree so a user can inspect exactly how the answer was built.
LangGraph-style orchestration, hybrid retrieval, and iterative validation are core parts of the design.
