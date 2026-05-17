# RESEARCH: ART-R for HAMH-RAG

## 1. Abstract
Traditional Retrieval-Augmented Generation (RAG) systems often rely on static query decomposition. When a sub-question fails due to insufficient evidence or contradictory sources, these systems either stall or perform naive retries. **ART-R** is a novel agentic architecture that treats the reasoning path as a dynamic, mutable tree. When validation fails, the system performs a structural "self-healing" operation by re-decomposing the failed node into more granular, verifiable sub-tasks.

## 2. Key Innovations

### A. Dynamic Tree Restructuring
Unlike static "Plan-and-Execute" models, ART-R monitors the health of each reasoning node. If a node (e.g., "What is the market cap of Company X?") fails terminal validation, the **Restructurer Agent** mutates the tree. It might split the question into "Identify the latest stock price of Company X" and "Identify the number of outstanding shares for Company X," effectively bypassing the data-point bottleneck.

### B. Source Consensus Coefficient (SCC)
ART-R introduces a cross-backend auditing mechanism. By comparing unstructured data (Vector DB) against structured facts (Knowledge Graph), the system calculates an **SCC score**. 
- **SCC ≈ 1.0:** High agreement between semantic and factual sources.
- **SCC < 0.5:** High conflict; triggers a specialized **Conflict Auditor** loop to resolve the contradiction before generation.

### C. Category-Aware Validation (CAV)
Traditional validators only check if an answer is "grounded" in context. ART-R introduces **CAV**, which extracts the intended **Entity Category** from the user's query (e.g., Country, Date, Currency). 
- If a user asks for a **Country** but the evidence only supports a **Region** (e.g., "Bohemia" instead of "Czech Republic"), CAV lowers the confidence and fails the node.
- This failure triggers a **Recursive Hop**, forcing the system to search for the modern mapping or containment relationship, solving the "Hidden Hop" problem where the answer is technically grounded but contextually incomplete.

## 3. Architecture Overview
- **Decomposer:** Builds the initial logic tree.
- **Retriever:** Hybrid Vector + Graph lookup.
- **Validator:** Calculates SCC and verifies groundedness.
- **Corrector:** Performs semantic query refinement.
- **Restructurer (Novel):** Mutates the tree structure on terminal failure.

## 4. Preliminary Results (Projected)
- **Resolution Rate:** Projected 25% improvement on complex multi-hop benchmarks (e.g., HotpotQA) where initial premises are often fuzzy.
- **Interpretability:** The final "Logic Tree" provides a full audit trail of how the system "changed its mind" and restructured its reasoning path.
