from hamhrag.agents.decomposer import QueryDecomposer
from hamhrag.agents.validator import AnswerValidator
from hamhrag.models import RetrievedDocument

# Test decomposer heuristics (no LLM needed)
d = QueryDecomposer(llm_client=None)

tests = [
    "When is Martha Bulloch Roosevelt's husband's birthday?",
    "Lendley C. Black is Chancellor of a university offering how many majors?",
    "Where did the father of Maria Brontë spend most of his adult life?",
    "What record label does the singer in 4 non blondes sign?",
    "Who was born first George Marshall or Allan Dwan?",
]
print("=== DECOMPOSER ===")
for q in tests:
    node = d.decompose(q)
    children = [c.question for c in node.children] if node.children else ["(no decomp — leaf)"]
    print(f"\nQ: {q}")
    for c in children:
        print(f"  -> {c}")

# Test validator fix for birthday
print("\n\n=== VALIDATOR ===")
v = AnswerValidator(llm_client=None)

fake_docs = [
    RetrievedDocument(source_id="d1", source_type="vector",
        content="Theodore Roosevelt Sr. was born on September 22, 1831.", score=0.9),
    RetrievedDocument(source_id="d2", source_type="vector",
        content="Martha Bulloch married Theodore Roosevelt Sr.", score=0.8),
]

validation_tests = [
    ("When is Martha Bulloch Roosevelt's husband's birthday?",
     "September 22, 1831", True),
    ("When was Theodore Roosevelt born?",
     "October 27, 1858", True),
    ("When was Theodore Roosevelt born?",
     "Theodore Roosevelt Jr. was born on October 27, 1858 in New York.", True),
]

for q, ans, expected_pass in validation_tests:
    res = v.validate(ans, fake_docs, question=q)
    status = "✓" if res.passed == expected_pass else "✗ WRONG"
    print(f"{status} Q: {q[:60]}")
    print(f"   Pass={res.passed} (expected {expected_pass}), Conf={res.confidence:.2f}, {res.rationale}")
