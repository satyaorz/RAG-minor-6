import json, os, random
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.pipeline import HamhRagPipeline

load_dotenv()
settings = HamhRagSettings.from_env()

pipeline = HamhRagPipeline(settings=settings)

# Pick specific queries we know we want to test boolean/multi-hop on
test_queries = [
    ("2wiki", "Who is the mother of the director of film Blind Spot (1932 Film)?", "Carrie Daumery"),
    ("2wiki", "Does Mario Beaulieu (Senator) have the same nationality as Ebenezer Porter?", "no"),
    ("2wiki", "Are director of film Vai Pandal and director of film Under Electric Clouds from the same country?", "no"),
    ("musique", "Who mothered the hostess of the party where Chopin met George Sand?", "Cosima Wagner"),
]

def cb(**kwargs):
    ev = kwargs.get("event", "")
    if ev == "node_start":
        print(f"  [{kwargs.get('node_id')}] {kwargs.get('question')}")
    elif ev == "node_done":
        s = "✅" if kwargs.get("status") == "verified" else "⚠️"
        print(f"  {s} [{kwargs.get('node_id')}] → {str(kwargs.get('answer',''))[:80]}")

correct = 0
total = len(test_queries)
for ds, q, expected in test_queries:
    print(f"\n{'='*70}")
    print(f"[{ds}] Q: {q}")
    print(f"Expected: {expected}")
    res = pipeline.run(q, progress_callback=cb, max_seconds=150.0)
    answer = res.final_answer or ""
    answer_clean = answer.split("Sources:")[0].strip().lower()
    expected_lower = expected.lower()
    match = expected_lower in answer_clean
    status = "✅ CORRECT" if match else "❌ WRONG"
    correct += match
    print(f"\n{status}")
    print(f"Answer: {answer[:120]}")
    print(f"Status: {res.root.status}")

print(f"\n{'='*70}")
print(f"SCORE: {correct}/{total} ({100*correct/total:.0f}%)")
