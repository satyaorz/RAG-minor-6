# Diagram 3: Latency Timeline

## Latency Timeline: Sequential vs Parallel Execution

Shows the time progression of sequential execution (before) vs parallel execution (after).

- **Yellow** = Sequential latency (slow, cumulative)
- **Green** = Parallel latency (fast, concurrent)
- **Blue** = Final output
- **Gray** = Performance metrics

```mermaid
graph LR
    subgraph SEQ["Sequential Execution (Before)"]
        direction TB
        S1["⏱ 0ms"]
        S2["Graph: 0-500ms"]
        S3["Check: 500-510ms"]
        S4["Vector: 510-1010ms"]
        S5["⏱ 1010ms Total"]
        
        S1 --> S2 --> S3 --> S4 --> S5
        style S2 fill:#FFC107,stroke:#F57F17
        style S3 fill:#FFEB3B,stroke:#F57C0C
        style S4 fill:#FFC107,stroke:#F57F17
        style S5 fill:#f44336,stroke:#c62828,color:#fff
    end
    
    subgraph PAR["Parallel Execution (After)"]
        direction TB
        P1["⏱ 0ms"]
        P2["Graph: 0-500ms"]
        P3["Vector: 0-500ms"]
        P4["Merge: 500-510ms"]
        P5["⏱ 510ms Total"]
        
        P1 --> P2 --> P4
        P1 --> P3 --> P4
        P4 --> P5
        style P2 fill:#4CAF50,stroke:#2E7D32,color:#fff
        style P3 fill:#4CAF50,stroke:#2E7D32,color:#fff
        style P5 fill:#2196F3,stroke:#1565C0,color:#fff
    end
    
    subgraph METRICS["Performance Gain"]
        M1["⏱ Time saved: 500ms"]
        M2["📊 Speedup: 1.98x"]
        M3["🎯 Queries affected: ~70%"]
        M4["💾 Memory: Same"]
    end
    
    SEQ -.->|Optimization Applied| PAR
    PAR -.-> METRICS
    
    style SEQ fill:#ffebee,stroke:#c62828
    style PAR fill:#e8f5e9,stroke:#2e7d32
    style METRICS fill:#e3f2fd,stroke:#1565C0
```

### Timeline Breakdown

**Sequential (Before)**:
1. 0-500ms: Graph search runs
2. 500-510ms: Results checked for entity tokens
3. 510-1010ms: Vector search runs as fallback (sequential!)
4. Total: **1010ms**

**Parallel (After)**:
1. 0-500ms: Graph and Vector search run simultaneously
2. 500-510ms: Results are merged and ranked
3. Total: **510ms**

**Savings**: 500ms per query (50% improvement)
