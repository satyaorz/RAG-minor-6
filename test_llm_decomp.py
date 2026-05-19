import os
import json
from treeqa.agents.decomposer import QueryDecomposer
from treeqa.backends.llm import LLMClient

client = LLMClient()
d = QueryDecomposer(llm_client=client)

query = "Which film has the director who was born earlier, People To Each Other or Tali-Ihantala 1944?"

heuristic = d._heuristic_decompose(query)
print("Heuristic:", heuristic)

llm_result = d._decompose_with_llm(query)
print("LLM directly:", llm_result)

sanitized = d._sanitize_questions(llm_result)
print("Sanitized:", sanitized)

