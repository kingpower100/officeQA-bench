# Data Directory

Benchmark datasets, processed chunks, embedding caches, FAISS indexes, and run outputs are intentionally not tracked in Git.

Expected local paths:

- `data/raw/documents.jsonl`
- `data/raw/questions_only.jsonl`
- `data/raw/qa_test.jsonl`
- `data/raw/ground_truth_contexts.jsonl`
- `data/processed/`
- `data/runs/`
- `data/eval/runs/`

Keep only lightweight documentation or `.gitkeep` placeholders under `data/` in the repository.
