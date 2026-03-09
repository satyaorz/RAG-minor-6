# LangGraph and Agentic Workflow Orchestration

## What Is LangGraph?

LangGraph is a library for building stateful, graph-structured multi-agent workflows on top of
LangChain. It represents an agent workflow as a directed graph where nodes are computation steps
(tool calls, LLM invocations, decision points) and edges encode control flow. Unlike linear
chains, LangGraph supports cycles — enabling retry loops, conditional branching, and iterative
reasoning patterns essential for complex agentic tasks.

## Core Concepts

### State

LangGraph maintains a typed state object that is passed through the graph and updated at each
node. State is the single shared data structure for the entire workflow — nodes read from it and
write to it. Using TypedDict or Pydantic models for state makes type errors easy to catch.

### Nodes

Nodes are Python callables that receive the current state and return a state update. They can:
- call an LLM
- execute a tool
- branch based on conditions
- transform data

### Edges

Edges define the transitions between nodes. LangGraph supports:
- **Static edges**: always transition from node A to node B.
- **Conditional edges**: inspect state and route to different nodes dynamically.

### Checkpointing

LangGraph can checkpoint state at each node boundary. This allows workflows to be paused,
resumed, inspected, or replayed without re-running completed steps. Checkpointing is essential
for long-running agentic tasks and human-in-the-loop patterns.

## Why LangGraph for RAG?

Traditional RAG pipelines are stateless and linear. LangGraph enables:

- **Multi-hop reasoning loops**: keep retrieving until evidence is sufficient.
- **Validation-driven retries**: if a sub-answer fails validation, re-decompose or re-retrieve.
- **Parallel subgraph execution**: fan-out sub-questions and join their answers.
- **Human approval gates**: pause before final answer when confidence is low.

## Comparison with LangChain Chains

LangChain chains execute linearly. LangGraph adds:
- Cycles and loops.
- Per-step state persistence.
- Fine-grained conditional routing.
- Explicit graph structure that is inspectable and debuggable.

## Agent Patterns in LangGraph

### ReAct (Reason + Act)

The ReAct pattern alternates between reasoning (LLM decides what to do next) and acting (tool
call). LangGraph models this as a loop between a "reason" node and an "act" node, with a
termination condition.

### Plan-and-Execute

A planner node generates a structured plan of steps. Executor nodes carry out each step. A
reviewer node checks completion. LangGraph's conditional edges make this straightforward.

### Reflection

A reflection node critiques the current answer and flags gaps. The graph loops back to retrieval
if the critique identifies missing evidence. This pattern aligns closely with TreeQA's validation
and correction step.
