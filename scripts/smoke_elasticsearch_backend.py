from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any


VECTOR_FIELD = "embedding"
TEXT_FIELD = "text"
EXPECTED_TOP_CHUNK_ID = "target"


@dataclass(frozen=True)
class SmokeChunk:
    chunk_id: str
    document_id: str
    original_context_id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Elasticsearch dense-vector retrieval.")
    parser.add_argument("--host", default=os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200"))
    parser.add_argument("--index-name", default=os.getenv("ELASTICSEARCH_SMOKE_INDEX"))
    parser.add_argument("--dense-dim", type=int, default=int(os.getenv("ELASTICSEARCH_DENSE_DIM", "3")))
    parser.add_argument("--username", default=os.getenv("ELASTICSEARCH_USERNAME"))
    parser.add_argument("--password", default=os.getenv("ELASTICSEARCH_PASSWORD"))
    parser.add_argument("--api-key", default=os.getenv("ELASTICSEARCH_API_KEY"))
    parser.add_argument("--verify-certs", action="store_true")
    parser.add_argument("--mode", choices=["script_score", "knn", "both"], default="both")
    parser.add_argument("--keep-index", action="store_true")
    return parser.parse_args(argv)


def build_synthetic_chunks(dense_dim: int) -> list[SmokeChunk]:
    if dense_dim < 2:
        raise ValueError("--dense-dim must be at least 2 for the smoke vectors.")
    target = [0.0] * dense_dim
    target[0] = 1.0
    second_axis = [0.0] * dense_dim
    second_axis[1] = 1.0
    weak_match = [0.0] * dense_dim
    weak_match[0] = 0.2
    weak_match[1] = 0.98
    opposite = [0.0] * dense_dim
    opposite[0] = -1.0
    return [
        SmokeChunk(
            chunk_id=EXPECTED_TOP_CHUNK_ID,
            document_id="doc_target",
            original_context_id="ctx_target",
            text="The expected smoke-test chunk.",
            metadata={"source_file": "synthetic_target.txt"},
            embedding=target,
        ),
        SmokeChunk(
            chunk_id="orthogonal",
            document_id="doc_orthogonal",
            original_context_id="ctx_orthogonal",
            text="A deliberately orthogonal chunk.",
            metadata={"source_file": "synthetic_orthogonal.txt"},
            embedding=second_axis,
        ),
        SmokeChunk(
            chunk_id="weak_match",
            document_id="doc_weak",
            original_context_id="ctx_weak",
            text="A weaker match with a small first component.",
            metadata={"source_file": "synthetic_weak.txt"},
            embedding=weak_match,
        ),
        SmokeChunk(
            chunk_id="opposite",
            document_id="doc_opposite",
            original_context_id="ctx_opposite",
            text="An opposite direction chunk.",
            metadata={"source_file": "synthetic_opposite.txt"},
            embedding=opposite,
        ),
    ]


def index_body(dense_dim: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "original_context_id": {"type": "keyword"},
                TEXT_FIELD: {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
                VECTOR_FIELD: {
                    "type": "dense_vector",
                    "dims": dense_dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }


def script_score_search_kwargs(index_name: str, query_vector: list[float], size: int) -> dict[str, Any]:
    return {
        "index": index_name,
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{VECTOR_FIELD}') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "source": ["chunk_id", "document_id", "original_context_id", TEXT_FIELD, "metadata"],
    }


def knn_search_kwargs(index_name: str, query_vector: list[float], size: int, num_candidates: int) -> dict[str, Any]:
    return {
        "index": index_name,
        "size": size,
        "knn": {
            "field": VECTOR_FIELD,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(num_candidates, size),
        },
        "source": ["chunk_id", "document_id", "original_context_id", TEXT_FIELD, "metadata"],
    }


def build_client(args: argparse.Namespace):
    try:
        from elasticsearch import Elasticsearch
    except Exception as ex:
        raise RuntimeError("The 'elasticsearch' package is required. Install project requirements first.") from ex
    kwargs: dict[str, Any] = {
        "request_timeout": 60,
        "verify_certs": bool(args.verify_certs),
    }
    if args.api_key:
        kwargs["api_key"] = args.api_key
    elif args.username is not None or args.password is not None:
        kwargs["basic_auth"] = (args.username or "", args.password or "")
    return Elasticsearch(args.host, **kwargs)


def create_index(client: Any, index_name: str, dense_dim: int) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=index_body(dense_dim))


def bulk_index_chunks(client: Any, index_name: str, chunks: list[SmokeChunk]) -> None:
    operations: list[dict[str, Any]] = []
    for chunk in chunks:
        operations.append({"index": {"_index": index_name, "_id": chunk.chunk_id}})
        operations.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "original_context_id": chunk.original_context_id,
                TEXT_FIELD: chunk.text,
                "metadata": chunk.metadata,
                VECTOR_FIELD: chunk.embedding,
            }
        )
    response = client.bulk(operations=operations, refresh=True)
    if response.get("errors"):
        raise RuntimeError(f"Bulk indexing returned errors: {response}")
    client.indices.refresh(index=index_name)


def run_search(client: Any, mode: str, index_name: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    start = time.perf_counter()
    if mode == "script_score":
        response = client.search(**script_score_search_kwargs(index_name, query_vector, top_k))
    elif mode == "knn":
        response = client.search(**knn_search_kwargs(index_name, query_vector, top_k, num_candidates=100))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    latency_ms = (time.perf_counter() - start) * 1000
    hits = response.get("hits", {}).get("hits", [])
    print(f"{mode}: hits={len(hits)} latency_ms={latency_ms:.2f}")
    return hits


def validate_hits(mode: str, hits: list[dict[str, Any]]) -> bool:
    if not hits:
        print(f"FAIL {mode}: expected non-empty results.")
        return False
    top_source = hits[0].get("_source") or {}
    top_chunk_id = str(top_source.get("chunk_id") or hits[0].get("_id"))
    if top_chunk_id != EXPECTED_TOP_CHUNK_ID:
        print(f"FAIL {mode}: expected top chunk '{EXPECTED_TOP_CHUNK_ID}', got '{top_chunk_id}'.")
        return False
    print(f"PASS {mode}: top chunk is '{top_chunk_id}'.")
    return True


def should_run_mode(requested: str, mode: str) -> bool:
    return requested == "both" or requested == mode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index_name = args.index_name or f"rag_benchmark_es_smoke_{int(time.time())}"
    chunks = build_synthetic_chunks(args.dense_dim)
    query_vector = chunks[0].embedding
    client = None
    ok = True
    try:
        client = build_client(args)
        info = client.info()
        cluster_name = info.get("cluster_name", "unknown")
        version = (info.get("version") or {}).get("number", "unknown")
        print(f"Connected to Elasticsearch cluster={cluster_name} version={version} host={args.host}")
        create_index(client, index_name, args.dense_dim)
        bulk_index_chunks(client, index_name, chunks)
        count = client.count(index=index_name).get("count")
        print(f"Indexed synthetic chunks index={index_name} count={count}")

        if should_run_mode(args.mode, "script_score"):
            ok = validate_hits("script_score", run_search(client, "script_score", index_name, query_vector, top_k=3)) and ok

        if should_run_mode(args.mode, "knn"):
            try:
                ok = validate_hits("knn", run_search(client, "knn", index_name, query_vector, top_k=3)) and ok
            except Exception as ex:
                if args.mode == "knn":
                    print(f"FAIL knn: {ex}")
                    ok = False
                else:
                    print(f"SKIP knn: native kNN is unavailable or rejected by this cluster: {ex}")
        if ok:
            print("PASS Elasticsearch smoke validation completed.")
            return 0
        print("FAIL Elasticsearch smoke validation failed.")
        return 1
    except Exception as ex:
        print(f"FAIL Elasticsearch smoke validation setup failed: {ex}")
        return 2
    finally:
        if client is not None and not args.keep_index:
            try:
                if client.indices.exists(index=index_name):
                    client.indices.delete(index=index_name)
                    print(f"Deleted temporary index {index_name}")
            except Exception as ex:
                print(f"WARN cleanup failed for index {index_name}: {ex}")
        elif args.keep_index:
            print(f"Kept index {index_name}")


if __name__ == "__main__":
    sys.exit(main())
