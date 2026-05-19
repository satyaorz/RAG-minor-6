"""End-to-end diagnostic: where does each benchmark query ACTUALLY fail?"""
import json, sys
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.backends.vector import build_vector_backend
from hamhrag.backends.graph import build_graph_backend
from hamhrag.backends.llm import build_llm_client
from hamhrag.agents.decomposer import QueryDecomposer
from hamhrag.retrieval import HybridRetriever

load_dotenv()
settings = HamhRagSettings.from_env()
print(f"vector_provider = {settings.vector_provider}")
print(f"graph_provider  = {settings.graph_provider}")
print(f"llm_provider    = {settings.llm_provider}")
print(f"max_retries     = {settings.max_retries}")
print(f"tree_retries    = {settings.tree_retries}")

vec = build_vector_backend(settings)
print(f"Vector backend  = {type(vec).__name__}")
if hasattr(vec, 'documents') and isinstance(vec.documents, list):
    print(f"  docs in index = {len(vec.documents)}")

graph = build_graph_backend(settings)
print(f"Graph backend   = {type(graph).__name__}")

llm = build_llm_client(settings)
decomposer = QueryDecomposer(llm_client=llm)
retriever = HybridRetriever(
    vector_backend=vec, graph_backend=graph, top_k=settings.retrieval_top_k
)

# Test queries from all 3 datasets
queries = [
    ("2wiki", "When is Martha Bulloch Roosevelt's husband's birthday?", "September 22, 1831"),
    ("hotpot", "Lendley C. Black is Chancellor of a university offering how many majors?", "85"),
    ("musique", "What record label does the singer in 4 non blondes sign?", "Custard Records"),
    ("hotpot", "Where did the father of Maria Brontë spend most of his adult life?", "England"),
]

print("\n" + "="*80)
for ds, q, expected in queries:
    print(f"\n[{ds}] Q: {q}")
    print(f"Expected: {expected}")

    # Step 1: Decompose
    tree = decomposer.decompose(q)
    subs = [c.question for c in tree.children] if tree.children else [tree.question]
    print(f"  Decomposed: {subs}")

    # Step 2: Retrieve for each sub-question
    for i, sub_q in enumerate(subs):
        docs = retriever.retrieve(sub_q)
        print(f"  Sub-{i+1}: {sub_q}")
        print(f"    Retrieved {len(docs)} docs:")
        for d in docs[:3]:
            print(f"      [{d.source_type} {d.score:.2f}] {d.source_id}: {d.content[:100]}")
        if not docs:
            print(f"      *** NO DOCS FOUND ***")
    print("-"*80)
