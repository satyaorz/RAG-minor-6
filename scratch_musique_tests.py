import json, os, random
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.pipeline import HamhRagPipeline

load_dotenv()
settings = HamhRagSettings.from_env()

pipeline = HamhRagPipeline(settings=settings)

def load_queries(filename, limit=5):
    queries = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                queries.append((data['question'], data['answer']))
            except:
                pass
    random.shuffle(queries)
    return queries[:limit]

# Get 5 queries
q_musique = load_queries('data/benchmark/musique_train_sample.jsonl', 5)

test_queries = [("musique", q, a) for q, a in q_musique]

def cb(**kwargs):
    ev = kwargs.get("event", "")
    if ev == "node_start":
        print(f"  [{kwargs.get('node_id')}] {kwargs.get('question')}")
    elif ev == "node_done":
        s = "✅" if kwargs.get("status") == "verified" else "⚠️"
        print(f"  {s} [{kwargs.get('node_id')}] → {str(kwargs.get('answer',''))[:80]}")
    elif ev == "decomposed":
        print(f"  🔀 Decomposed into: {' | '.join(kwargs.get('sub_questions', []))}")

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
