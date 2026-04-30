# Diagram 2: Before vs After Optimization

## Optimization: Before vs After - Eliminating Sequential Bottleneck

Visual comparison of sequential execution (before optimization) vs parallel execution (after optimization).

- **Red** = Sequential execution (slow ⛔)
- **Green** = Parallel execution (fast ✅)
- **Blue** = Final results

```mermaid
graph LR
    subgraph BEFORE["❌ BEFORE (7x slowdown)"]
        A1["Query: 'Explain who founded Apple?'"] --> B1["graph_first route chosen<br/>(threshold: 0.28)"]
        B1 --> C1["Run Graph Search<br/>~500ms"]
        C1 --> D1["Check: graph_docs have entities?"]
        D1 -->|NO - entity tokens missing| E1["Run Vector Search<br/>~500ms"]
        D1 -->|YES| F1["Return"]
        E1 --> F1
        style C1 fill:#FFC107,stroke:#F57F17,color:#000
        style E1 fill:#FFC107,stroke:#F57F17,color:#000
        style F1 fill:#f44336,stroke:#c62828,color:#fff
    end
    
    subgraph AFTER["✅ AFTER (50% faster)"]
        A2["Query: 'Explain who founded Apple?'"] --> B2["QueryRouter analyzes<br/>hybrid signals"]
        B2 --> C2{_predict_graph_fallback_<br/>needed?}
        C2 -->|YES - mixed intent detected| D2["Use hybrid_parallel<br/>Parallel Execution"]
        C2 -->|NO - pure lookup| E2["Sequential graph_first<br/>~500ms"]
        D2 --> F2["Vector + Graph<br/>simultaneously ⚡<br/>Latency = max~500ms"]
        F2 --> G2["Return Results"]
        E2 --> G2
        style D2 fill:#4CAF50,stroke:#2E7D32,color:#fff
        style F2 fill:#4CAF50,stroke:#2E7D32,color:#fff
        style G2 fill:#2196F3,stroke:#1565C0,color:#fff
    end
    
    BEFORE -.->|Threshold raised 0.28→0.35| AFTER
    
    L1["Timeline: 0—500ms—1000ms"] -.-> T1["BEFORE: 500ms + 500ms = 1000ms ⛔"]
    L1 -.-> T2["AFTER: max500ms, 500ms = 500ms ✅"]
    
    style BEFORE fill:#ffebee,stroke:#c62828,color:#000
    style AFTER fill:#e8f5e9,stroke:#2e7d32,color:#000
    style T1 fill:#f44336,stroke:#c62828,color:#fff
    style T2 fill:#4CAF50,stroke:#2E7D32,color:#fff
```

### Key Insights

- **Before**: Queries like "Explain who founded Apple?" were routed to `graph_first` (threshold 0.28), then when graph results were weak, vector search was triggered sequentially (~1000ms total)
- **After**: The same query now either stays sequential (if pure lookup) or switches to parallel execution early (~500ms total)
- **Threshold Change**: 0.28 → 0.35 makes the router more conservative, pushing borderline queries to parallel execution
- **Speedup**: ~1.98x faster (50% latency reduction)
