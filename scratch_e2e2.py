"""Full end-to-end benchmark test v2 — with improved prompt + retry settings."""
import json, os, importlib
from hamhrag.config import HamhRagSettings, load_dotenv

# Force re-read of .env
for k in list(os.environ.keys()):
    if k.startswith("HAMHRAG_"):
        del os.environ[k]
load_dotenv()

settings = HamhRagSettings.from_env()
print(f"max_retries={settings.max_retries} tree_retries={settings.tree_retries} top_k={settings.retrieval_top_k}")

from hamhrag.pipeline import HamhRagPipeline
pipeline = HamhRagPipeline(settings=settings)

queries = [
    ("2wiki", "When is Martha Bulloch Roosevelt's husband's birthday?", "September 22, 1831"),
    ("hotpot", "Lendley C. Black is Chancellor of a university offering how many majors?", "85"),
    ("musique", "What record label does the singer in 4 non blondes sign?", "Custard Records"),
    ("hotpot", "Where did the father of Maria Brontë spend most of his adult life?", "England"),
    ("2wiki", "What nationality is the director of film Kadvi Hawa?", "India"),
    ("hotpot", "What is the name of the architect who designed the Lincoln Memorial dedicated in 1922?", "Henry Bacon"),
    ("musique", "When did the person who said \"I think, therefore I am\" live?", "1596-1650"),
]

def cb(**kwargs):
    ev = kwargs.get("event", "")
    if ev == "node_start":
        print(f"  [{kwargs.get('node_id')}] {kwargs.get('question')}")
    elif ev == "node_done":
        s = "✅" if kwargs.get("status") == "verified" else "⚠️"
        print(f"  {s} [{kwargs.get('node_id')}] → {str(kwargs.get('answer',''))[:80]}")

correct = 0
total = len(queries)
for ds, q, expected in queries:
    print(f"\n{'='*70}")
    print(f"[{ds}] Q: {q}")
    print(f"Expected: {expected}")
    res = pipeline.run(q, progress_callback=cb, max_seconds=90.0)
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
