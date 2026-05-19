"""Test: what does the pipeline ACTUALLY produce for hop-1?"""
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.backends.vector import build_vector_backend
from hamhrag.backends.graph import build_graph_backend
from hamhrag.backends.llm import build_llm_client
from hamhrag.agents import AnswerGenerator, AnswerValidator
from hamhrag.retrieval import HybridRetriever

load_dotenv()
settings = HamhRagSettings.from_env()
vec = build_vector_backend(settings)
graph = build_graph_backend(settings)
llm = build_llm_client(settings)
retriever = HybridRetriever(vector_backend=vec, graph_backend=graph, top_k=settings.retrieval_top_k)
generator = AnswerGenerator(llm_client=llm)
validator = AnswerValidator(llm_client=llm)

# Sub-question 1 for each benchmark query
sub_questions = [
    "Who is Martha Bulloch Roosevelt's husband?",
    "What university is Lendley C. Black the Chancellor of?",
    "Who is the singer in 4 non blondes?",
    "Who is the father of Maria Brontë?",
]

for sq in sub_questions:
    docs = retriever.retrieve(sq)
    print(f"\nQ: {sq}")
    print(f"  Retrieved {len(docs)} docs")
    answer = generator.generate_for_node(sq, docs)
    print(f"  Generated answer: {answer[:120]}")
    val = validator.validate(answer, docs, question=sq)
    print(f"  Validation: passed={val.passed} conf={val.confidence:.2f} | {val.rationale[:80]}")
