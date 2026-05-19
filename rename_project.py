import os
import re

def rename_content(content, file_path):
    # 1. Environment variables: TREEQA_ -> HAMHRAG_
    content = content.replace("TREEQA_", "HAMHRAG_")
    
    # 2. Case-insensitive whole word 'treeqa' (except citations)
    # We use word boundaries to avoid matching 'zhang2025treeqa'
    
    def replace_word(match):
        word = match.group(0)
        if word == "treeqa":
            return "hamhrag"
        if word == "TREEQA":
            return "HAMHRAG"
        if word == "TreeQA":
            return "HamhRag"
        return "hamhrag" # fallback

    content = re.sub(r'\btreeqa\b', 'hamhrag', content)
    content = re.sub(r'\bTREEQA\b', 'HAMHRAG', content)
    
    # Surgical replacement for TreeQA to HamhRag (especially for classes)
    content = re.sub(r'\bTreeQA\b', 'HamhRag', content)
    
    # 3. Handle specific path segments in strings (e.g. "src/treeqa/ui")
    content = content.replace("src/treeqa", "src/hamhrag")
    
    # 4. Handle "treeqa.cli" etc (dotted access)
    content = content.replace("treeqa.cli", "hamhrag.cli")
    content = content.replace("treeqa.api", "hamhrag.api")
    content = content.replace("treeqa.ui", "hamhrag.ui")
    
    # 5. Handle "HAMH-RAG" title if needed (optional, but good for docs)
    # The user said rename from treeqa to hamhrag.
    
    return content

# List of files to process
include_dirs = ['src/hamhrag', 'tests', 'data/benchmark']
include_files = [
    'pyproject.toml', 'Makefile', 'README.md', 'BENCHMARK.md', 
    'PROGRESS.txt', 'OPTIMIZATION.md', '.env.example', '.env', 
    'project_spec.md', 'test_llm_decomp.py'
]

for root_file in include_files:
    if os.path.exists(root_file):
        print(f"Processing {root_file}...")
        with open(root_file, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = rename_content(content, root_file)
        if new_content != content:
            with open(root_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

for d in include_dirs:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.html', '.json', '.jsonl')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read()
                    except UnicodeDecodeError:
                        continue
                new_content = rename_content(content, file_path)
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

print("Done.")
