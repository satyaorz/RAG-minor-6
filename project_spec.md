# Project TreeQA: RAG Hallucination Detection and Mitigation

## 1. The Aim

**Primary Objective:** Build an advanced, hallucination-resistant RAG system optimized for Multi-Hop Question Answering (MHQA) by implementing a hierarchical, self-correcting reasoning framework.

**Specific Goals:**
* Enhance interpretability by visualizing the exact reasoning path (the "Logic Tree") the model takes.
* Detect hallucinations actively during each step of a complex query.
* Mitigate errors dynamically through an iterative self-correction mechanism before final output generation.

---

## 2. The Approach

Adopt an agentic or graph-based RAG architecture (using LangGraph or LlamaIndex Workflows) based on four main pillars:

### A. Query Decomposition (The Tree Builder)
Use an LLM to decompose a complex user query into a tree of simpler, verifiable sub-questions. 

### B. Hybrid Information Retrieval
Minimize hallucination by cross-referencing unstructured data (Vector DBs like Pinecone or Qdrant for Wikipedia/documents) with structured data (Knowledge Graphs like Wikidata or Neo4j for verifiable facts).

### C. Hallucination Detection (Iterative Validation)
Evaluate each sub-question node using Natural Language Inference (NLI) or an "LLM-as-a-judge" to verify if the retrieved context fully supports the generated sub-answer, assigning a confidence score.

### D. Mitigation (Self-Correction Engine)
Halt forward propagation if a sub-answer is flagged as a hallucination. Prompt the system to modify its search query, switch databases, or admit a lack of knowledge rather than fabricating an answer.

---

## 3. The Project Plan (Roadmap)

### Phase 1: Research and Setup (Weeks 1-2)
* Locate and analyze the open-source TreeQA code mentioned in the research paper.
* Set up LangChain and LangGraph for workflow orchestration.
* Select LLMs (e.g., GPT-4o, Claude 3.5, or local Llama-3).
* Provision databases (Neo4j for graph data, Qdrant/ChromaDB for vector data).
* Download benchmark datasets like HotpotQA or 2WikiMultiHopQA for testing.

### Phase 2: Building Core Modules (Weeks 3-5)
* Build the Decomposer Prompt to output structured JSON trees of sub-queries.
* Develop the Hybrid Retriever functions to route queries to appropriate databases.
* Create the Validator module to output Pass/Fail scores based on context and sub-answers.

### Phase 3: Integration and Self-Correction Loop (Weeks 6-7)
* Connect all modules using LangGraph to create a cyclic workflow.
* Program the logic loop to move forward on Pass or trigger self-correction/re-retrieval on Fail.
* Aggregate verified sub-answers into the final response.

### Phase 4: Evaluation and Metrics (Week 8)
* Benchmark the pipeline against a standard "Naive RAG" baseline.
* Measure Faithfulness, Answer Relevance, and Context Precision using RAGAS or TruLens.

### Phase 5: UI and Visualization (Weeks 9-10)
* Build a user interface using Streamlit or Gradio.
* Implement a visualizer (using streamlit-agraph or Mermaid.js) to display the "Logic Tree" to the user.
* Highlight verified nodes in green and self-corrected nodes in yellow to ensure transparent reasoning.