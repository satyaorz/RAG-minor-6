# Multi-Hop Question Answering

## Overview

Multi-hop question answering (QA) requires a system to chain together multiple pieces of evidence
from different sources to produce a single coherent answer. Unlike single-hop QA, where one
passage contains the full answer, multi-hop QA involves reasoning across two or more documents or
facts. This makes it significantly harder for retrieval-augmented generation (RAG) systems.

## Key Datasets

### HotpotQA

HotpotQA is a large-scale multi-hop QA dataset with over 113,000 question-answer pairs sourced
from English Wikipedia. Questions require reasoning over two Wikipedia paragraphs. The dataset
provides supporting facts at the sentence level, which allows systems to be evaluated on both
answer extraction and supporting fact prediction. HotpotQA defines two types of questions:
bridge questions (where entity A connects to entity B) and comparison questions (where two
entities are compared on a shared property).

### 2WikiMultiHopQA

2WikiMultiHopQA is a multi-hop QA benchmark that draws from two Wikipedia-style sources and
requires reasoning across evidence chains of up to four hops. It defines five question types:
bridge, inference, comparison, temporal, and compositional. Unlike HotpotQA, it provides logical
reasoning chains as explicit annotations, making it better suited to measuring structured
multi-hop reasoning.

### MuSiQue

MuSiQue is a multi-hop QA dataset that enforces strict decomposability. Each question is
constructed by composing simpler single-hop questions, and annotators verify that direct answers
cannot be found without performing all hops. This makes MuSiQue a strong test of genuine
compositional reasoning.

## Retrieval Strategies

Effective multi-hop QA requires retrieval to follow the reasoning chain. Common strategies include:

- **Iterative retrieval**: retrieve documents, extract bridging entities, then retrieve more.
- **Query decomposition**: break the question into sub-questions and retrieve for each.
- **Dense passage retrieval (DPR)**: use learned embedding models to retrieve relevant passages.
- **Graph-augmented retrieval**: model entity relationships as a graph and traverse it.

## Evaluation Metrics

Multi-hop QA is typically evaluated with:
- **Exact Match (EM)**: whether the predicted answer string exactly matches the gold answer.
- **F1 score**: token-level overlap between the predicted answer and the gold answer.
- **Supporting fact F1**: whether the system retrieves the correct supporting sentences.
- **Joint F1**: combined score over answer F1 and supporting fact F1.
