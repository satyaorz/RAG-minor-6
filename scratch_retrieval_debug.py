"""Debug: what does the retriever actually return for the failing hop-2 questions?"""
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.backends.vector import build_vector_backend
from hamhrag.backends.graph import build_graph_backend
from hamhrag.backends.llm import build_llm_client
from hamhrag.agents import AnswerGenerator
from hamhrag.retrieval import HybridRetriever

load_dotenv()
settings = HamhRagSettings.from_env()
vec = build_vector_backend(settings)
graph = build_graph_backend(settings)
llm = build_llm_client(settings)
retriever = HybridRetriever(vector_backend=vec, graph_backend=graph, top_k=6)
generator = AnswerGenerator(llm_client=llm)

# These are the contextualized hop-2 queries after entity substitution
hop2_queries = [
    ("Roosevelt bday", "When is Theodore Roosevelt's birthday? Theodore Roosevelt"),
    ("Linda Perry label", "What record label does Linda Perry sign? Linda Perry"),
    ("Patrick Bronte life", "Where did Patrick Brontë spend most of his adult life? Patrick Brontë"),
    ("Lincoln Memorial", "Who is the architect that designed the Lincoln Memorial dedicated in 1922?"),
    ("Descartes dates", "When did René Descartes live? René Descartes"),
]

for label, q in hop2_queries:
    docs = retriever.retrieve(q)
    print(f"\n{'='*60}")
    print(f"[{label}] Q: {q}")
    for d in docs[:4]:
        print(f"  [{d.source_type} {d.score:.2f}] {d.source_id}")
        print(f"    {d.content[:150]}")
    
    if docs:
        answer = generator.generate_for_node(q, docs)
        print(f"  ANSWER: {answer[:100]}")
