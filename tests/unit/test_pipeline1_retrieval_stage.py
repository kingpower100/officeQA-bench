import numpy as np

from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.retrieval_stage import RetrievalStage


def test_retrieval_stage_returns_retrieval_item_compatible_outputs():
    cfg = _cfg()
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]

    output = RetrievalStage(cfg, _Embedder(), _FaissIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert output.attempted == 1
    assert row.query.question_id == "q1"
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c2"]
    assert [item.chunk_id for item in row.retrieved] == ["c1"]
    assert row.retrieved[0].original_context_id == "ctx-c1"
    assert row.retrieved[0].metadata["document_id"] == "doc-c1"
    assert row.retrieval_time_ms >= 0


def test_retrieval_stage_preserves_raw_vs_final_ids_after_dedupe():
    cfg = _cfg(fetch_k=2, top_k=2)
    chunks = [_chunk("c1", "alpha"), _chunk("c1", "alpha duplicate")]

    output = RetrievalStage(cfg, _Embedder(), _DuplicateIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c1"]
    assert [item.chunk_id for item in row.retrieved] == ["c1"]
    assert row.retrieval_warnings == ["Only 1 unique chunks were available after deduplication; requested top_k=2."]


def test_retrieval_stage_reranker_enabled_path_still_works():
    cfg = _cfg(top_k=2, fetch_k=2, reranker_enabled=True, final_top_k=1)
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]

    output = RetrievalStage(
        cfg,
        _Embedder(),
        _FaissIndex(),
        chunks,
        reranker_factory=lambda model_name, device: _ReverseReranker(),
    ).run(StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]}))

    row = output.retrieval_rows[0]
    assert output.final_top_k == 1
    assert row.reranker_used is True
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c2"]
    assert [item.chunk_id for item in row.retrieved] == ["c2"]
    assert row.retrieved[0].rerank_score == 1.0


def test_retrieval_stage_elasticsearch_dense_retriever_works():
    cfg = _cfg(retriever_type="elasticsearch_dense", top_k=1, fetch_k=2)
    chunks = [_chunk("c1", "alpha")]

    output = RetrievalStage(cfg, _Embedder(), _ElasticsearchDenseIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert row.retrieved[0].chunk_id == "c1"
    assert row.retrieved[0].retrieval_source == "elasticsearch_dense"
    assert row.retrieved[0].dense_score == 0.75
    assert row.raw_dense_retrieved[0].chunk_id == "c1"


class _Embedder:
    def encode_query(self, question):
        return np.ones(2, dtype="float32")


class _FaissIndex:
    def search(self, query_embedding, top_k):
        return np.array([1.0, 0.9], dtype="float32")[:top_k], np.array([0, 1], dtype="int64")[:top_k]


class _DuplicateIndex:
    def search(self, query_embedding, top_k):
        return np.array([1.0, 0.9], dtype="float32")[:top_k], np.array([0, 1], dtype="int64")[:top_k]


class _ElasticsearchDenseIndex:
    text_field = "text"

    def search_hits(self, query_vec, candidate_k):
        return [
            {
                "_id": "c1",
                "_score": 1.75,
                "_source": {
                    "chunk_id": "c1",
                    "document_id": "doc-c1",
                    "original_context_id": "ctx-c1",
                    "text": "alpha",
                    "metadata": {"document_id": "doc-c1", "file_name": "c1.txt"},
                },
            }
        ]


class _ReverseReranker:
    requested_device = "cpu"
    runtime_device = "cpu"

    def rerank(self, question, items, top_k):
        reranked = []
        for score, item in enumerate(reversed(items), start=1):
            reranked.append(item.model_copy(update={"score": float(score), "rerank_score": float(score)}))
        return reranked[:top_k]


def _chunk(chunk_id: str, text: str):
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"ctx-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"document_id": f"doc-{chunk_id}", "file_name": f"{chunk_id}.txt"},
    )


def _cfg(
    retriever_type: str = "dense",
    top_k: int = 1,
    fetch_k: int = 2,
    reranker_enabled: bool = False,
    final_top_k: int | None = None,
):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": retriever_type, "top_k": top_k, "fetch_k": fetch_k},
            "reranker": {
                "enabled": reranker_enabled,
                "model_name": "fake-reranker" if reranker_enabled else None,
                "device": "cpu",
                "final_top_k": final_top_k,
            },
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )
