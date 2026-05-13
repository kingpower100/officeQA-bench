import os
import json
from pathlib import Path

import requests


def _resolve_path(base_dir: Path | None, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() or base_dir is None else (base_dir / path).resolve()


def run_preflight_checks(cfg, base_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    docs_path = _resolve_path(base_dir, cfg.data.documents_path)
    qa_path = _resolve_path(base_dir, cfg.data.qa_test_path)
    for name, path in [("documents_path", docs_path), ("qa_test_path", qa_path)]:
        if not path.exists() or not path.is_file():
            errors.append(f"{name} is missing or not a file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{name} is empty: {path}")
    if cfg.retrieval.fetch_k < cfg.retrieval.top_k:
        errors.append(f"retrieval.fetch_k ({cfg.retrieval.fetch_k}) must be >= retrieval.top_k ({cfg.retrieval.top_k})")
    if cfg.chunking.chunk_overlap >= cfg.chunking.chunk_size:
        errors.append(f"chunking.chunk_overlap ({cfg.chunking.chunk_overlap}) must be < chunking.chunk_size ({cfg.chunking.chunk_size})")
    if cfg.index.metric == "cosine" and not cfg.embedding.normalize_embeddings:
        errors.append("embedding.normalize_embeddings must be true when index.metric is cosine")
    if cfg.embedding.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("embedding.device is cuda but CUDA is not available to torch")
        except Exception as ex:
            errors.append(f"embedding.device is cuda but torch/CUDA could not be checked: {ex}")
    if qa_path.exists() and qa_path.is_file():
        errors.extend(_validate_question_ids(qa_path, cfg.data.question_id_field))
    if os.getenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "0") != "1":
        base_url = os.getenv("OLLAMA_BASE_URL", cfg.generation.base_url).rstrip("/")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=min(cfg.generation.timeout_s, 10))
            response.raise_for_status()
            available_models = _ollama_model_names(response.json())
            if cfg.generation.model_name not in available_models:
                available = ", ".join(sorted(available_models)) or "<none>"
                errors.append(f"Ollama model '{cfg.generation.model_name}' not found at {base_url}/api/tags. Available: {available}")
        except requests.RequestException as ex:
            errors.append(f"Unable to reach Ollama at {base_url}/api/tags: {ex}")
    return errors


def _validate_question_ids(path: Path, question_id_field: str) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as ex:
                errors.append(f"questions file has invalid JSON on line {line_number}: {ex}")
                continue
            question_id = row.get(question_id_field)
            if question_id is None:
                continue
            question_id = str(question_id)
            if question_id in seen:
                duplicates.add(question_id)
            seen.add(question_id)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        errors.append(f"questions file contains duplicate question IDs in field '{question_id_field}': {sample}")
    return errors


def _ollama_model_names(payload: dict) -> set[str]:
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if value:
                names.add(str(value))
    return names
