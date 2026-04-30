# Diagram 6: Complete Data Flow

## End-to-End Data Flow: Question → Answer

Shows the complete journey of a user question through the TreeQA system, from decomposition through retrieval to final answer with reasoning.

- **Green (⚡)** = Optimized retrieval routing
- **Blue** = Decomposition and initial processing
- **Red** = Validation and reasoning agents
- **Purple** = User input/output
- **Orange** = Data retrieval

```mermaid
graph TB
    USER["👤 User Query<br/>e.g. 'Who founded Apple?'"]
    
    USER --> API["📡 API<br/>FastAPI / Streamlit / CLI"]
    API --> DECOMP["🔄 Query Decomposition<br/>(LLM)<br/>Breaks into sub-questions"]
    
    DECOMP --> TREE_BUILD["🌳 Build Logic Tree<br/>Root: Main question<br/>Children: Sub-questions"]
    
    TREE_BUILD --> RESOLVE["🔍 Resolve Each Node<br/>for each sub-question"]
    
    RESOLVE --> ROUTER["🛣️ Query Router<br/>(OPTIMIZED)<br/>Analyzes intent"]
    
    ROUTER --> ROUTE_DECISION{Route Selection}
    
    ROUTE_DECISION -->|Graph First| GRAPH_BE["📊 Graph Backend<br/>LocalGraphBackend<br/>Neo4j"]
    ROUTE_DECISION -->|Vector First| VEC_BE["🔬 Vector Backend<br/>FAISS/LocalVectorBackend<br/>Qdrant"]
    ROUTE_DECISION -->|Hybrid Parallel| HYBRID["⚡ Parallel Execution<br/>ThreadPoolExecutor<br/>Both: graph + vector"]
    
    GRAPH_BE --> DATA_GRAPH["📄 Graph Data<br/>facts.jsonl<br/>Entity relationships"]
    VEC_BE --> DATA_VEC["📚 Vector Index<br/>FAISS Index<br/>Embeddings"]
    HYBRID --> DATA_GRAPH
    HYBRID --> DATA_VEC
    
    GRAPH_BE --> DOCS["📖 Retrieved Docs<br/>(Graph source)"]
    VEC_BE --> DOCS
    
    DOCS --> RANK["🏆 Rank & Merge<br/>RRF Scoring<br/>Top-K Selection"]
    
    RANK --> VAL["✅ Validate<br/>(LLM)<br/>Grounding check<br/>Entity alignment"]
    
    VAL --> VAL_CHECK{Valid?}
    VAL_CHECK -->|YES| GEN["📝 Generate Answer<br/>(LLM)<br/>Synthesize from evidence"]
    VAL_CHECK -->|NO| CORR["🔧 Correct Query<br/>(LLM)<br/>Refine + retry"]
    
    CORR --> RETRY_DECOMP["♻️ Retry Decomposition"]
    RETRY_DECOMP --> RESOLVE
    
    GEN --> STRUCT["🎯 Structure Result<br/>Confidence score<br/>Evidence links<br/>Rationale"]
    
    STRUCT --> RESTR["🔄 Restructure Tree<br/>(LLM)<br/>Optional re-arrangement"]
    
    RESTR --> OUTPUT["📊 Output<br/>JSON + Logic Tree<br/>Visualization"]
    
    OUTPUT --> RENDER["🎨 Render UI<br/>FastAPI SPA<br/>Streamlit dashboard"]
    
    RENDER --> USER_VIEW["👁️ User Views<br/>- Interactive tree<br/>- Retrieval scores<br/>- Evidence docs<br/>- Confidence metrics"]
    
    style ROUTER fill:#4CAF50,stroke:#2E7D32,color:#fff,stroke-width:2px
    style HYBRID fill:#4CAF50,stroke:#2E7D32,color:#fff
    style ROUTE_DECISION fill:#FF9800,stroke:#E65100,color:#fff
    style DECOMP fill:#2196F3,stroke:#1565C0,color:#fff
    style VAL fill:#F44336,stroke:#C62828,color:#fff
    style GEN fill:#F44336,stroke:#C62828,color:#fff
    style CORR fill:#F44336,stroke:#C62828,color:#fff
    style USER fill:#9C27B0,stroke:#6A1B9A,color:#fff
    style USER_VIEW fill:#9C27B0,stroke:#6A1B9A,color:#fff
    style DATA_GRAPH fill:#FF5722,stroke:#BF360C,color:#fff
    style DATA_VEC fill:#FF5722,stroke:#BF360C,color:#fff
```

### Data Flow Stages

**1. Input & Decomposition**
- User asks question via API (FastAPI/Streamlit/CLI)
- LLM decomposes into logical sub-questions
- Build reasoning tree with root (main) and child nodes

**2. Node Resolution** (for each node)
- Each node is resolved independently
- Intelligent routing analyzes query intent
- Selects optimal retrieval strategy

**3. Retrieval** (⚡ OPTIMIZED)
- QueryRouter analyzes intent signals
- Routes to: vector_first, vector_only, graph_first, or hybrid_parallel
- For graph_first: predicts if fallback needed
- Executes backends (sequential or parallel)
- Retrieves top-K documents

**4. Ranking & Merging**
- Combines results from all backends
- RRF scoring for fair ranking
- Removes duplicates

**5. Validation** (Agent: Validator)
- Checks answer grounding in evidence
- Verifies entity alignment
- Evaluates confidence

**6. Self-Correction Loop** (Agent: Corrector)
- If validation fails:
  - LLM refines the query
  - Decomposer re-runs
  - Process repeats (up to max retries)
- Implements hallucination detection

**7. Answer Generation** (Agent: Generator)
- LLM synthesizes final answer
- Cites evidence and reasoning
- Structures with confidence scores

**8. Tree Restructuring** (Agent: Restructurer)
- Optional: Reorganizes logic tree
- Improves clarity
- Consolidates insights

**9. Output & Visualization**
- Returns JSON with:
  - Answer text
  - Confidence score
  - Evidence documents
  - Logic tree structure
  - Retrieval traces (routing info)
- Renders in UI:
  - Interactive tree visualization
  - Retrieval score charts
  - Evidence panels
  - Confidence metrics

### Key Optimization Point

The **retrieval stage** (⚡ green) is where the optimization applies:
- Previously: Sequential execution when fallback triggered (~1000ms)
- Now: Parallel execution for mixed-intent queries (~500ms)
- Impact: 50% latency reduction for ~70% of queries
