from __future__ import annotations

import json
import os
import shutil
import time
import subprocess
from pathlib import Path

from src.pipeline1.chunking.fixed_token_chunker import FixedTokenChunker
from src.pipeline1.chunking.fixed_word_chunker import FixedWordChunker
from src.pipeline1.chunking.sentence_chunker import SentenceChunker
from src.pipeline1.embedding.cache import EmbeddingCache
from src.pipeline1.embedding.factory import build_embedder
from src.pipeline1.generation.cost_estimator import estimate_cost
from src.pipeline1.generation.factory import build_generator
from src.pipeline1.generation.prompt_builder import PROMPT_TEMPLATE_VERSION, build_prompt, dedupe_prompt_contexts
from src.pipeline1.indexing.factory import build_index
from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.io.manifest_writer import write_manifest
from src.pipeline1.io.result_writer import ResultWriter
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.output_record import OutputRecord
from src.pipeline1.telemetry.logger import build_logger
from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict
from src.pipeline1.utils.seed import set_seed
from tqdm.auto import tqdm


def run_pipeline(config_path: str) -> Path:
    start_time = time.time()
    print("[1/10] Loading config")
    cfg = PipelineConfig.from_yaml(config_path)
    set_seed(cfg.experiment.random_seed)
    project_root = _project_root()
    run_dir = project_root / cfg.experiment.output_dir / cfg.experiment.experiment_id

    if run_dir.exists() and cfg.runtime.overwrite:
        for name in ("results.jsonl", "results.csv", "run_manifest.json", "logs.txt", "pipeline1.log"):
            path = run_dir / name
            if path.exists():
                path.unlink()
    elif run_dir.exists() and not cfg.runtime.resume and (run_dir / "results.jsonl").exists():
        raise FileExistsError(f"Run already exists and resume=false: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir / "logs.txt", cfg.runtime.log_level)

    print("[2/10] Running preflight checks")
    preflight_errors = run_preflight_checks(cfg, project_root)
    if preflight_errors:
        raise RuntimeError("; ".join(preflight_errors))
    docs_path = project_root / cfg.data.documents_path
    qa_path = project_root / cfg.data.qa_test_path
    print("[3/10] Loading documents")
    docs = JsonlReader.read_documents(str(docs_path), require_context_id=True)

    cache_dir = project_root / "data" / "processed"
    chunks_key = stable_hash_dict(
        {
            "documents_sha256": file_sha256(docs_path),
            "chunking": cfg.chunking.model_dump(),
        }
    )
    chunks_path = cache_dir / "chunks" / f"{chunks_key}.jsonl"
    print("[4/10] Chunking documents")
    chunks = _load_chunks(chunks_path)
    if chunks is None:
        chunks = _build_chunker(cfg).chunk_documents(docs, show_progress=True)
        _save_chunks(chunks_path, chunks)
    else:
        logger.info("Loaded cached chunks: %s", chunks_path)

    embedder = build_embedder(cfg.embedding)
    embeddings_key = stable_hash_dict(
        {
            "chunks_key": chunks_key,
            "embedding": cfg.embedding.model_dump(),
        }
    )
    embeddings_path = cache_dir / "embeddings" / f"{embeddings_key}.npy"
    print("[5/10] Generating embeddings")
    embeddings = EmbeddingCache.load(embeddings_path)
    if embeddings is None:
        embeddings = embedder.encode_texts([chunk.text for chunk in chunks], show_progress=True)
        EmbeddingCache.save(embeddings_path, embeddings, {"chunks_key": chunks_key, "embedding": cfg.embedding.model_dump()})
    else:
        logger.info("Loaded cached embeddings: %s", embeddings_path)

    index_key = stable_hash_dict(
        {
            "embeddings_key": embeddings_key,
            "index": cfg.index.model_dump(),
        }
    )
    index_path = cache_dir / "indexes" / f"{index_key}.faiss"
    index = build_index(cfg.index)
    print("[6/10] Building/loading FAISS index")
    if index_path.exists():
        index.load(str(index_path))
        logger.info("Loaded FAISS index: %s", index_path)
    else:
        index.build(embeddings)
        index.save(str(index_path))
        logger.info("Built FAISS index: %s", index_path)

    print("[7/10] Loading questions")
    queries = list(JsonlReader.iter_queries(str(qa_path), cfg.data.question_id_field, cfg.data.question_field, logger))
    _log_run_info(logger, cfg, docs_count=len(docs), chunk_count=len(chunks), question_count=len(queries), qa_path=qa_path)

    retriever = build_retriever(cfg.retrieval, embedder, index, chunks)
    reranker = CrossEncoderReranker(cfg.reranker.model_name) if cfg.reranker.enabled and cfg.reranker.model_name else None
    generator = build_generator(cfg.generation)
    writer = ResultWriter(run_dir, save_csv=cfg.runtime.save_csv, logger=logger)
    existing_ids = writer.load_existing_question_ids() if cfg.runtime.resume else set()
    pending_queries = [query for query in queries if query.question_id not in existing_ids]

    attempted = 0
    written = 0
    retrieval_rows = []
    try:
        print("[8/10] Retrieving contexts")
        for index, query in enumerate(tqdm(pending_queries, desc="Retrieving contexts", unit="question"), start=1):
            attempted += 1
            logger.info(
                "row_start phase=retrieval question_id=%s row=%s/%s",
                query.question_id,
                index,
                len(pending_queries),
            )
            retrieval_start = time.perf_counter()
            if reranker is None:
                raw_retrieved = retriever.retrieve(query.question, cfg.retrieval.fetch_k)
                reranker_used = False
            else:
                candidates = retriever.retrieve(query.question, cfg.retrieval.fetch_k)
                raw_retrieved = reranker.rerank(query.question, candidates, cfg.retrieval.fetch_k)
                reranker_used = True
            retrieved = dedupe_retrieval_by_original_context_id(raw_retrieved, cfg.retrieval.top_k)
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            logger.info(
                "row_retrieved question_id=%s raw_candidates=%s unique_final_contexts=%s scores=%s retrieval_time_ms=%.2f",
                query.question_id,
                len(raw_retrieved),
                len(retrieved),
                len([item.score for item in retrieved]),
                retrieval_time_ms,
            )
            retrieval_rows.append((query, retrieved, retrieval_time_ms, reranker_used))

        print("[9/10] Generating answers")
        for index, (query, retrieved, retrieval_time_ms, reranker_used) in enumerate(
            tqdm(retrieval_rows, desc="Generating answers", unit="question"), start=1
        ):
            prompt_contexts = dedupe_prompt_contexts(retrieved)
            logger.info(
                "row_start phase=generation question_id=%s row=%s/%s saved_contexts=%s prompt_contexts=%s",
                query.question_id,
                index,
                len(retrieval_rows),
                len(retrieved),
                len(prompt_contexts),
            )
            prompt = build_prompt(cfg.generation.system_prompt, query.question, prompt_contexts)
            generation_start = time.perf_counter()
            error = None
            try:
                generation = generator.generate(prompt)
                answer = generation.answer
                input_tokens = generation.input_tokens
                output_tokens = generation.output_tokens
            except Exception as ex:
                answer = ""
                input_tokens = 0
                output_tokens = 0
                error = str(ex)
                logger.exception("row_generation_error question_id=%s error=%s", query.question_id, error)
            generation_time_ms = (time.perf_counter() - generation_start) * 1000
            total_tokens = input_tokens + output_tokens
            cost = (
                estimate_cost(
                    input_tokens,
                    output_tokens,
                    cfg.telemetry.pricing.input_per_1k_tokens_usd,
                    cfg.telemetry.pricing.output_per_1k_tokens_usd,
                )
                if cfg.telemetry.estimate_cost
                else 0.0
            )

            record = OutputRecord(
                experiment_id=cfg.experiment.experiment_id,
                question_id=query.question_id,
                question=query.question,
                generated_answer=answer,
                retrieved_chunk_ids=[item.chunk_id for item in retrieved],
                retrieved_original_context_ids=[item.original_context_id for item in retrieved],
                retrieved_context_ids=[item.original_context_id for item in retrieved],
                retrieved_chunk_texts=[item.text for item in retrieved],
                retrieved_context_texts=[item.text for item in retrieved],
                retrieval_scores=[item.score for item in retrieved],
                top_k=cfg.retrieval.top_k,
                chunking_strategy=cfg.chunking.strategy,
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
                embedding_model=cfg.embedding.model_name,
                retriever_type=cfg.retrieval.retriever_type,
                reranker_used=reranker_used,
                llm_model=cfg.generation.model_name,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_latency_ms=retrieval_time_ms + generation_time_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                error=error,
            )
            writer.write(record)
            written += 1
            logger.info(
                "row_written question_id=%s answer_chars=%s input_tokens=%s output_tokens=%s total_latency_ms=%.2f error=%s",
                query.question_id,
                len(answer),
                input_tokens,
                output_tokens,
                retrieval_time_ms + generation_time_ms,
                error,
            )

        print("[10/10] Writing outputs")
    finally:
        writer.close()

    resolved_config = cfg.model_dump()
    resolved_config["generation"]["base_url"] = os.getenv("OLLAMA_BASE_URL", cfg.generation.base_url)
    end_time = time.time()
    output_counts = _output_row_counts(run_dir)
    write_manifest(
        run_dir,
        {
            "config_path": str(Path(config_path).resolve()),
            "config_hash": file_sha256(config_path),
            "config": cfg.model_dump(),
            "resolved_config": resolved_config,
            "data_hashes": {
                "documents_path": str(docs_path),
                "documents_sha256": file_sha256(docs_path),
                "questions_path": str(qa_path),
                "questions_sha256": file_sha256(qa_path),
            },
            "cache_keys": {"chunks": chunks_key, "embeddings": embeddings_key, "index": index_key},
            "output_row_counts": output_counts,
            "run_stats": {"n_documents": len(docs), "n_queries": len(queries), "attempted": attempted, "written": written},
            "pipeline_version": "0.1.0",
            "git_commit": _git_commit(project_root),
            "start_timestamp_utc": _iso_utc(start_time),
            "end_timestamp_utc": _iso_utc(end_time),
        },
    )
    return run_dir


def dedupe_retrieval_by_original_context_id(items: list, top_k: int) -> list:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = str(item.original_context_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def _log_run_info(
    logger,
    cfg: PipelineConfig,
    docs_count: int,
    chunk_count: int | None,
    question_count: int,
    qa_path: Path,
) -> None:
    reranker_state = "enabled" if cfg.reranker.enabled else "disabled"
    logger.info("experiment_id=%s", cfg.experiment.experiment_id)
    logger.info("document_count=%s", docs_count)
    logger.info("chunk_count=%s", chunk_count if chunk_count is not None else "pending")
    logger.info("question_count=%s", question_count)
    logger.info("embedding_model=%s", cfg.embedding.model_name)
    logger.info("embedding_device=%s", cfg.embedding.device)
    logger.info("generator_model=%s", cfg.generation.model_name)
    logger.info("top_k=%s", cfg.retrieval.top_k)
    logger.info("fetch_k=%s", cfg.retrieval.fetch_k)
    logger.info("reranker=%s", reranker_state)
    if cfg.reranker.enabled and cfg.reranker.model_name:
        logger.info("reranker_model=%s", cfg.reranker.model_name)
    logger.info("question_input_path=%s", qa_path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_chunker(cfg: PipelineConfig):
    if cfg.chunking.strategy == "fixed_token":
        return FixedTokenChunker(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap, cfg.chunking.tokenizer_name)
    if cfg.chunking.strategy == "fixed_word":
        return FixedWordChunker(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    print("WARNING: chunking.strategy='sentence' currently uses whitespace word windows; it is not true sentence-aware.")
    return SentenceChunker(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)


def _load_chunks(path: Path) -> list[ChunkRecord] | None:
    if not path.exists():
        return None
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(ChunkRecord.model_validate_json(line))
    return chunks


def _save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def _output_row_counts(run_dir: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for name in ("results.jsonl", "results.csv"):
        path = run_dir / name
        if not path.exists():
            counts[name] = None
            continue
        with path.open("r", encoding="utf-8") as f:
            row_count = sum(1 for line in f if line.strip())
        counts[name] = max(0, row_count - 1) if name.endswith(".csv") and row_count else row_count
    return counts


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _iso_utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
