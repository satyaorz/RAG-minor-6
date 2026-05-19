from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.backends.vector import LocalVectorBackend
from hamhrag.models import RetrievedDocument

load_dotenv()
settings = HamhRagSettings()
backend = LocalVectorBackend(settings=settings)

queries = [
    "Theodore Roosevelt Sr birthday",
    "Martha Bulloch Roosevelt husband",
    "Lendley C. Black Chancellor university majors",
    "singer 4 non blondes Linda Perry",
]

for q in queries:
    docs = backend.search(q, limit=3)
    print(f"\nQ: {q}")
    if docs:
        for d in docs:
            print(f"  [{d.score:.2f}] {d.source_id}: {d.content[:120]}")
    else:
        print("  NO RESULTS")
