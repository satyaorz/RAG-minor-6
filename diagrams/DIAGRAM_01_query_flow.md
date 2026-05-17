# Diagram 1: Query Flow After Optimization

## HAMH-RAG Route Optimization: Query Flow After Optimization

Shows how queries are routed through the intelligent routing system with fallback prediction.

- **Green (⚡)** = Parallel execution paths (optimized)
- **Orange** = Graph-first path with prediction
- **Blue** = Input/Output

```mermaid
graph TD
    A["Query Input"] --> B["QueryRouter.route"]
    B --> C{Analyze Query}
    C -->|Multi-hop/Comparison| D1["hybrid_parallel<br/>(Both backends in parallel)"]
    C -->|Strong Entity/Relation<br/>graph_score - vector_score ≥ 0.35| E1["graph_first"]
    C -->|Simple Descriptive<br/>vector_score >> graph_score| F1["vector_first"]
    C -->|Ambiguous| D1
    
    E1 --> E2{Predict Fallback<br/>Needed?}
    E2 -->|YES| E3["Use hybrid_parallel<br/>(Parallel execution)"]
    E2 -->|NO| E4["Sequential:<br/>Graph first"]
    
    F1 --> F2["Run Vector<br/>Check Confidence"]
    F2 -->|Needs Fallback| F3["Run Graph"]
    F2 -->|Sufficient| F4["Return Results"]
    
    D1 --> D2["Run Vector & Graph<br/>in Parallel<br/>⚡ Latency = max(t_v, t_g)"]
    
    E3 --> D2
    E4 --> E5["Run Graph"]
    E5 -->|Low Confidence| E6["Run Vector"]
    E5 -->|High Confidence| E7["Return Results"]
    E6 --> E7
    
    F3 --> F4
    D2 --> D3["Merge & Rank"]
    E7 --> D3
    D3 --> H["Final Results"]
    
    style D2 fill:#4CAF50,stroke:#2E7D32,color:#fff
    style E3 fill:#4CAF50,stroke:#2E7D32,color:#fff
    style E4 fill:#FFC107,stroke:#F57F17,color:#000
    style A fill:#2196F3,stroke:#1565C0,color:#fff
    style H fill:#2196F3,stroke:#1565C0,color:#fff
```
