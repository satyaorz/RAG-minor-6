from hamhrag.pipeline import HamhRagPipeline
from hamhrag.config import HamhRagSettings, load_dotenv
from hamhrag.agents.decomposer import QueryDecomposer

load_dotenv()
settings = HamhRagSettings(max_retries=1, max_restructures=1)
pipeline = HamhRagPipeline(settings=settings)

queries = [
    ("2wiki", "When is Martha Bulloch Roosevelt's husband's birthday?", "September 22, 1831"),
    ("hotpot", "Lendley C. Black is Chancellor of a university offering how many majors?", "85"),
    ("hotpot", "Where did the father of Maria Brontë spend most of his adult life?", "England"),
    ("musique", "What record label does the singer in 4 non blondes sign?", "Custard Records"),
]

def cb(event, **kwargs):
    if event == "node_start":
        print(f"  START [{kwargs.get('node_id')}]: {kwargs.get('question')}")
    elif event == "node_done":
        print(f"  DONE  [{kwargs.get('node_id')}]: {kwargs.get('status')} | conf: {kwargs.get('confidence', '?')} | ans: {kwargs.get('answer', '')[:80]}")

for dataset, q, expected in queries:
    print(f"\n{'='*70}")
    print(f"[{dataset}] Q: {q}")
    print(f"Expected: {expected}")
    res = pipeline.run(q, progress_callback=cb)
    print(f"Got: {res.final_answer[:100] if res.final_answer else 'NONE'}")
    print(f"Root status: {res.root.status}")

