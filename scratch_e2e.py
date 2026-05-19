"""Full end-to-end benchmark test with real LLM + real vector index."""
import json
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.pipeline import HamhRagPipeline

load_dotenv()
settings = HamhRagSettings.from_env()

# Override retries to give the pipeline a fair chance
settings.max_retries = 2
settings.tree_retries = 1

pipeline = HamhRagPipeline(settings=settings)

# All benchmark queries
queries = [
    ("2wiki", "When is Martha Bulloch Roosevelt's husband's birthday?", "September 22, 1831"),
    ("hotpot", "Lendley C. Black is Chancellor of a university offering how many majors?", "85"),
    ("musique", "What record label does the singer in 4 non blondes sign?", "Custard Records"),
    ("hotpot", "Where did the father of Maria Brontë spend most of his adult life?", "England"),
    ("2wiki", "What nationality is the director of film Kadvi Hawa?", "India"),
    ("hotpot", "What is the name of the architect who designed the Lincoln Memorial dedicated in 1922?", "Henry Bacon"),
    ("musique", "When did the person who said \"I think, therefore I am\" live?", "1596-1650"),
]

def cb(event, **kwargs):
    if event == "node_start":
        print(f"  [{kwargs.get('node_id')}] {kwargs.get('question')}")
    elif event == "node_done":
        status_icon = "✅" if kwargs.get("status") == "verified" else "⚠️"
        print(f"  {status_icon} [{kwargs.get('node_id')}] → {str(kwargs.get('answer', ''))[:80]}")

correct = 0
total = len(queries)
for ds, q, expected in queries:
    print(f"\n{'='*70}")
    print(f"[{ds}] Q: {q}")
    print(f"Expected: {expected}")
    
    res = pipeline.run(q, progress_callback=cb, max_seconds=60.0)
    
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
