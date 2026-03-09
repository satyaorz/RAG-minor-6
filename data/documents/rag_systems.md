# Retrieval-Augmented Generation (RAG) Systems

## What Is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval system with a
language model to ground generated answers in external knowledge. Instead of relying solely on
parametric knowledge baked into model weights during training, RAG systems fetch relevant
documents at inference time and condition the language model on those documents.

The core benefit of RAG is reducing hallucination: the model can produce answers grounded in
retrieved passages rather than generating from memory alone.

## Basic RAG Pipeline

A standard RAG pipeline consists of three stages:

1. **Query encoding**: embed the user query using a dense encoder.
2. **Retrieval**: search a vector index for the most relevant passages.
3. **Generation**: condition the language model on the retrieved passages to produce an answer.

## Advanced RAG Patterns

### Iterative RAG

Iterative RAG performs multiple retrieval rounds. After a first pass, the system uses partial
answers or extracted entities to retrieve additional documents. This is especially effective for
multi-hop queries.

### Self-RAG

Self-RAG trains the language model to decide when to retrieve, and to critique its own outputs
against retrieved evidence. This reduces unnecessary retrieval and teaches the model to reject
poorly supported claims.

### Corrective RAG (CRAG)

Corrective RAG adds an evaluation step after retrieval. If retrieved documents score poorly on
relevance, the system falls back to web search or structured knowledge bases to find better
evidence before generating.

### Modular RAG

Modular RAG treats the pipeline as composable modules. Individual components such as retrieval,
reranking, summarization, and generation can be swapped or improved independently.

## Retrieval Methods

### Sparse Retrieval

BM25 and TF-IDF are classic sparse retrieval methods based on term frequency. They are fast and
interpretable but fail to capture semantic similarity when query and document wording differ.

### Dense Retrieval

Dense retrieval uses neural encoders (e.g., BERT-based bi-encoders) to project queries and
passages into a shared embedding space and compute cosine similarity. It handles paraphrase and
vocabulary mismatch better than sparse methods. DPR (Dense Passage Retrieval) and Contriever are
common examples.

### Hybrid Retrieval

Hybrid retrieval combines sparse (BM25) and dense scores, typically via reciprocal rank fusion or
a learned interpolation. It consistently outperforms either alone because it exploits complementary
signal from exact lexical match and semantic similarity.

### Reranking

Cross-encoders such as BGE-Reranker or Cohere Rerank take (query, passage) pairs and produce a
richer relevance score at higher computational cost. Reranking a larger first-stage candidate set
with a cross-encoder is a common way to improve precision.

## Common Hallucination Sources in RAG

- Retrieved documents that are irrelevant to the query.
- Documents that are relevant but outdated or conflicting.
- Model generating text beyond what is explicitly stated in retrieved evidence.
- Too few retrieved documents to cover all aspects of a multi-part question.

## Grounding and Citation

Grounded RAG systems require the language model to cite specific passages that support each
factual claim. Structured citation format (e.g., [source_id] after each claim) allows
post-hoc verification and builds user trust. Evidence linking — tying each answer sentence
back to a retrieved passage — is key to hallucination mitigation.
