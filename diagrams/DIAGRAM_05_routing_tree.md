# Diagram 5: Intelligent Routing Decision Tree

## Routing Decision Tree: How Queries Are Classified

Shows the complete decision tree used to classify queries and select optimal retrieval strategy.

- **Green (⚡)** = Parallel execution paths (optimized)
- **Orange** = Graph-first with fallback prediction
- **Blue** = Vector-first paths
- **Light green** = Input/Output

```mermaid
graph TD
    START["Question Input"] --> TOKENIZE["Tokenize & Extract Features"]
    
    TOKENIZE --> SCORES["Calculate Scores"]
    SCORES --> GS["graph_score = <br/>relation_hits × 0.3<br/>+ multi_hop × 0.35<br/>+ entity_hint × 0.12<br/>+ w/w/when × 0.08"]
    SCORES --> VS["vector_score = <br/>explanation_hits × 0.28<br/>+ not_multi_hop × 0.18<br/>+ long_query × 0.08<br/>+ explain/describe × 0.08"]
    
    GS --> DECISION{Decision Tree}
    VS --> DECISION
    
    DECISION -->|multi_hop OR<br/>comparison| D1["🔹 hybrid_parallel<br/>Both backends run<br/>in parallel"]
    
    DECISION -->|graph_score -<br/>vector_score ≥ 0.35| D2["🔹 graph_first"]
    
    DECISION -->|vector_score -<br/>graph_score ≥ 0.35<br/>AND short query| D3["🔹 vector_only<br/>Vector runs first"]
    
    DECISION -->|vector_score ><br/>graph_score| D4["🔹 vector_first<br/>Vector runs first"]
    
    DECISION -->|else| D1
    
    D2 --> FALLBACK["⚡ NEW: Predict Fallback"]
    FALLBACK --> FALLBACK_CHECK{will_fallback?}
    FALLBACK_CHECK -->|YES<br/>mixed intent| D2_PAR["Use hybrid_parallel<br/>(⚡ Parallel execution)"]
    FALLBACK_CHECK -->|NO<br/>pure lookup| D2_SEQ["Use graph_first<br/>(Sequential)"]
    
    D1 --> EXE1["ThreadPoolExecutor<br/>Vector + Graph<br/>Simultaneously"]
    D2_PAR --> EXE1
    D3 --> EXE2["Vector Search<br/>→ Fallback if needed"]
    D4 --> EXE2
    D2_SEQ --> EXE3["Graph Search<br/>→ Vector Fallback<br/>if confidence low"]
    
    EXE1 --> MERGE["Merge & Rank"]
    EXE2 --> MERGE
    EXE3 --> MERGE
    
    MERGE --> OUTPUT["Return Top-K<br/>with traces"]
    
    style D1 fill:#4CAF50,stroke:#2E7D32,color:#fff
    style D2_PAR fill:#4CAF50,stroke:#2E7D32,color:#fff
    style EXE1 fill:#4CAF50,stroke:#2E7D32,color:#fff
    style D2 fill:#FF9800,stroke:#E65100,color:#fff
    style FALLBACK fill:#FF9800,stroke:#E65100,color:#fff
    style FALLBACK_CHECK fill:#FF9800,stroke:#E65100,color:#fff
    style D3 fill:#2196F3,stroke:#1565C0,color:#fff
    style D4 fill:#2196F3,stroke:#1565C0,color:#fff
    style START fill:#e8f5e9,stroke:#2e7d32,color:#000
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,color:#000
```

### Decision Tree Logic

**Step 1: Feature Extraction**
- Extract tokens and analyze query structure
- Identify capitalized words (entities)
- Count keyword occurrences

**Step 2: Score Calculation**
- **graph_score**: How much the query needs structured data (entity lookups, relationships)
- **vector_score**: How much the query needs semantic understanding (explanations, descriptions)

**Step 3: Route Selection**
1. If multi-hop or comparison detected → **hybrid_parallel** (both at once)
2. If graph_score - vector_score ≥ 0.35 → **graph_first** (strong entity signals)
3. If vector_score - graph_score ≥ 0.35 AND short → **vector_only** (simple lookup)
4. If vector_score > graph_score → **vector_first** (descriptive)
5. Else → **hybrid_parallel** (ambiguous, use both)

**Step 4: Fallback Prediction** (NEW - for graph_first only)
- Predict if fallback will be needed WITHOUT running actual search
- If YES (mixed intent) → Switch to **hybrid_parallel** (parallel execution ⚡)
- If NO (pure lookup) → Keep **graph_first** (sequential execution)

**Step 5: Execution**
- Execute the selected route with appropriate backend(s)
- Merge results from all backends
- Rank using RRF (Reciprocal Rank Fusion)
- Return top-K results with retrieval traces

### Example Classifications

```
"Who founded Apple?"
├─ Tokens: [who, founded, apple]
├─ graph_score: 0.80 (strong relation + entity)
├─ vector_score: 0.18 (weak explanation)
├─ Route: graph_first
├─ Fallback prediction: NO
└─ Execution: Sequential graph search

"Explain how machine learning works"
├─ Tokens: [explain, machine, learning, works]
├─ graph_score: 0.12 (no relations)
├─ vector_score: 0.82 (strong explanation)
├─ Route: vector_first
└─ Execution: Vector search + fallback if needed

"Compare Apple and Microsoft founders"
├─ Tokens: [compare, apple, microsoft, founders]
├─ Multi-hop detected: YES (compare)
├─ Route: hybrid_parallel
└─ Execution: Parallel (both backends)
```
