from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.backends.vector import build_vector_backend

load_dotenv()
settings = HamhRagSettings()
backend = build_vector_backend(settings)

# Check what's in the index
print(f"Backend type: {type(backend).__name__}")
if hasattr(backend, 'documents'):
    print(f"Total docs in index: {len(backend.documents)}")
    # Sample 5 docs
    for doc in backend.documents[:5]:
        print(f"  - {doc.get('source_id', '?')}: {str(doc.get('content',''))[:80]}")

# Now test retrieval directly
queries = [
    "Martha Bulloch Roosevelt husband birthday",
    "Theodore Roosevelt Sr born September 1831",
    "Lendley C Black Chancellor university majors",
    "Linda Perry 4 Non Blondes singer record label",
]
print()
for q in queries:
    docs = backend.search(q, limit=3)
    print(f"Q: {q}")
    if docs:
        for d in docs:
            print(f"  [{d.score:.3f}] {d.source_id}: {d.content[:100]}")
    else:
        print("  NO RESULTS")
    print()
