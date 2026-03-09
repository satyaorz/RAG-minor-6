# Hallucination Detection and Evaluation

## What Is LLM Hallucination?

In the context of large language models, hallucination refers to generated text that is
factually incorrect, unsupported by retrieved evidence, or internally inconsistent. Hallucination
is a critical problem for RAG systems because users expect answers grounded in the retrieved
documents, not in the model's potentially stale or confabulated parametric knowledge.

## Types of Hallucination

### Intrinsic Hallucination

The model contradicts the retrieved source. For example, a passage says "X was founded in 1990"
but the model generates "X was founded in 1985".

### Extrinsic Hallucination

The model introduces information not present in any retrieved source. This is harder to detect
because the extra information may be true (but unverifiable from the given context) or false.

## Hallucination Mitigation Strategies

### Evidence Grounding

Require the model to cite specific passages and restrict generation to sourced claims only.
System prompts such as "Answer only from the provided evidence" and post-hoc citation checks
reduce extrinsic hallucination.

### LLM-as-Judge

Use a second LLM call to evaluate whether the generated answer is supported by the retrieved
passages. The judge receives the answer and evidence and returns a structured verdict
(passed/failed) with a confidence score and rationale. This is the approach used by TreeQA's
validator agent.

### Natural Language Inference (NLI)

NLI models take a (premise, hypothesis) pair and classify the relationship as entailment,
neutral, or contradiction. Applied to RAG, the premise is the retrieved passage and the
hypothesis is each claim in the generated answer. Claims classified as contradiction or neutral
(with no other supporting passage) are flagged as potential hallucinations.

### RAGAS Metrics

RAGAS (Retrieval-Augmented Generation Assessment) is an evaluation framework that measures:
- **Faithfulness**: fraction of claims in the answer attributable to the retrieved context.
- **Answer relevance**: how relevant the generated answer is to the question.
- **Context precision**: fraction of retrieved context that is relevant.
- **Context recall**: how much of the ground truth is covered by retrieved context.

### TruLens

TruLens is an evaluation tool for LLM applications. It records all inputs, outputs, and
intermediate steps of a pipeline and evaluates them with configurable feedback functions
including groundedness, answer relevance, and context relevance.

## Validation in RAG Pipelines

A validation step after generation checks whether the answer is grounded in retrieved evidence.
Common signals used for validation:
- Token overlap between answer and evidence.
- Semantic similarity between answer sentences and evidence passages.
- LLM judge verdict.
- NLI entailment score.

TreeQA's validator uses an LLM judge as the primary signal and lexical overlap as the fallback.
A validated sub-answer increases confidence; a failed validation triggers a retry with a refined query.
