from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.agents.decomposer import QueryDecomposer
from hamhrag.pipeline import HamhRagPipeline

load_dotenv()

# Step 1: See what decomposer produces
decomposer = QueryDecomposer(llm_client=None)  # heuristic only first
queries = [
    "When is Martha Bulloch Roosevelt's husband's birthday?",
    "Lendley C. Black is Chancellor of a university offering how many majors?",
    "Where does Lance Stephenson of the Indiana Pacers basketball team play his home games?",
    "What is the nationality of the singer-songwriter who wrote the poetry collection Early Work?",
]

for q in queries:
    node = decomposer.decompose(q)
    subs = [c.question for c in node.children]
    print(f"Q: {q}")
    print(f"  → {subs if subs else ['[single leaf]']}")
    print()

# Step 2: Check what documents we have for key entities
settings = HamhRagSettings()
pipeline = HamhRagPipeline(settings=settings)

test_queries = [
    "Lendley C. Black Chancellor university",
    "University of Minnesota Duluth majors",
    "Martha Bulloch Roosevelt husband",
    "Theodore Roosevelt Sr birthday 1831",
]
for q in test_queries:
    docs = pipeline.retriever.retrieve(q)
    print(f"Query: {q}")
    for d in docs:
        print(f"  [{d.source_id}] score={d.score:.2f}: {d.content[:100]}")
    print()
