# Diagram 4: System Architecture Overview

## Complete System Architecture: TreeQA with Optimized Routing

Shows the complete system components and how they interact, with the optimized routing highlighted in green.

- **Green (⚡)** = Optimized intelligent routing (NEW)
- **Blue** = Core pipeline components
- **Orange** = Storage backends
- **Purple** = Data & indices
- **Cyan** = User interfaces
- **Red** = Reasoning agents

```mermaid
graph TB
    subgraph INPUT["🔍 INPUT & INTERFACE"]
        UI1["FastAPI SPA<br/>index.html"]
        UI2["Streamlit UI"]
        CLI["CLI Interface"]
    end
    
    subgraph PIPELINE["🏗️ CORE PIPELINE"]
        DECOMP["QueryDecomposer<br/>(LLM)"]
        RETR["HybridRetriever"]
        VAL["AnswerValidator<br/>(LLM)"]
        CORR["CorrectionEngine<br/>(LLM)"]
        GEN["AnswerGenerator<br/>(LLM)"]
        RESTR["TreeRestructurer<br/>(LLM)"]
    end
    
    subgraph ROUTING["🛣️ INTELLIGENT ROUTING (OPTIMIZED)"]
        ROUTER["QueryRouter<br/>Threshold: 0.35"]
        PREDICT["_predict_graph_<br/>fallback_needed"]
        HYBRID["HybridRetriever<br/>parallel/sequential"]
    end
    
    subgraph BACKENDS["🗄️ STORAGE BACKENDS"]
        VEC["Vector Backend<br/>LocalVectorBackend<br/>FaissVectorBackend<br/>QdrantVectorBackend"]
        GRAPH["Graph Backend<br/>LocalGraphBackend<br/>Neo4jGraphBackend<br/>MemoryGraphBackend"]
        LLM["LLM Backend<br/>OpenAI/OpenRouter<br/>Caching"]
    end
    
    subgraph DATA["📊 DATA & INDICES"]
        DOCS["Document Corpus<br/>data/documents/"]
        FACTS["Fact Graph<br/>data/graph/facts.jsonl"]
        VINDEX["Vector Index<br/>.faiss + .jsonl"]
        BENCH["Benchmarks<br/>HotpotQA, MuSiQue"]
    end
    
    subgraph AGENTS["🤖 REASONING AGENTS"]
        RETR_A["Retrieval Agent"]
        VAL_A["Validation Agent<br/>(Grounding Check)"]
        CORR_A["Correction Agent<br/>(Self-healing)"]
        GEN_A["Generation Agent<br/>(Answer Synthesis)"]
    end
    
    INPUT --> PIPELINE
    
    DECOMP --> RETR
    RETR --> ROUTING
    ROUTING --> PREDICT
    PREDICT --> HYBRID
    HYBRID --> BACKENDS
    
    BACKENDS --> VEC
    BACKENDS --> GRAPH
    BACKENDS --> LLM
    
    DATA --> VEC
    DATA --> GRAPH
    
    PIPELINE --> VAL
    VAL --> CORR
    CORR --> GEN
    GEN --> RESTR
    
    VAL --> AGENTS
    CORR --> AGENTS
    GEN --> AGENTS
    
    RESTR --> OUTPUT["📈 OUTPUT<br/>JSON Result<br/>Logic Tree"]
    
    OUTPUT --> UI1
    OUTPUT --> UI2
    
    style ROUTING fill:#4CAF50,stroke:#2E7D32,color:#fff,stroke-width:3px
    style PREDICT fill:#4CAF50,stroke:#2E7D32,color:#fff
    style HYBRID fill:#4CAF50,stroke:#2E7D32,color:#fff
    style PIPELINE fill:#2196F3,stroke:#1565C0,color:#fff
    style BACKENDS fill:#FF9800,stroke:#E65100,color:#fff
    style DATA fill:#9C27B0,stroke:#6A1B9A,color:#fff
    style INPUT fill:#00BCD4,stroke:#00838F,color:#fff
    style OUTPUT fill:#00BCD4,stroke:#00838F,color:#fff
    style AGENTS fill:#F44336,stroke:#C62828,color:#fff
```

### Component Details

**Input Interfaces**:
- FastAPI SPA: Web-based interactive UI with live routing visualization
- Streamlit UI: Alternative dashboard interface
- CLI: Command-line interface for batch processing

**Core Pipeline**:
- QueryDecomposer: Breaks complex questions into sub-questions using LLM
- HybridRetriever: Fetches evidence from multiple backends
- AnswerValidator: Checks grounding and entity alignment
- CorrectionEngine: Self-healing loop for invalid answers
- AnswerGenerator: Synthesizes final answer from evidence
- TreeRestructurer: Optional tree reorganization

**Intelligent Routing** (⚡ Optimized):
- QueryRouter: Analyzes query intent and selects best route
- Fallback Prediction: Predicts if sequential execution will need fallback
- HybridRetriever: Executes parallel or sequential based on prediction

**Storage Backends**:
- Vector: Semantic similarity search (FAISS, Qdrant, Local)
- Graph: Structured fact retrieval (Neo4j, Local, Memory)
- LLM: Language model API calls with built-in caching

**Data Sources**:
- Document Corpus: Raw unstructured documents
- Fact Graph: Structured knowledge base
- Vector Index: Pre-computed embeddings
- Benchmarks: Evaluation datasets
