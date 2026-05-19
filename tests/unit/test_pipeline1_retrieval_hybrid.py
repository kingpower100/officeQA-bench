import pytest

from src.pipeline1.orchestrator import retrieve_top_k_unique_contexts
from src.pipeline1.retrieval.bm25_retriever import BM25Retriever
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.retrieval.hybrid_rrf_retriever import HybridRRFRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.retrieval import RetrievalItem


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"{chunk_id}.txt",
        original_context_id=f"{chunk_id}.txt",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"file_name": f"{chunk_id}.txt", "chunk_unit": "test"},
    )


def test_bm25_retriever_returns_lexical_matches():
    retriever = BM25Retriever(
        [
            _chunk("c1", "veterans administration expenditures public works"),
            _chunk("c2", "foreign claims and securities"),
        ]
    )

    rows = retriever.retrieve("veterans expenditures", top_k=2)

    assert rows[0].chunk_id == "c1"
    assert rows[0].retrieval_source == "bm25"
    assert rows[0].bm25_score is not None
    assert rows[0].dense_score is None


def test_hybrid_rrf_combines_dense_and_bm25_rankings():
    dense = _FakeDense(
        [
            _item("dense_only", 0.9),
            _item("shared", 0.8),
        ]
    )
    bm25 = _FakeBM25(
        [
            _item("bm25_only", 7.0, source="bm25"),
            _item("shared", 5.0, source="bm25"),
        ]
    )
    retriever = HybridRRFRetriever(dense, bm25, fetch_k=2, rrf_k=60, dense_weight=1.0, bm25_weight=1.0)

    rows = retriever.retrieve("query", top_k=3)

    assert {row.chunk_id for row in rows} == {"dense_only", "bm25_only", "shared"}
    shared = next(row for row in rows if row.chunk_id == "shared")
    assert shared.retrieval_source == "hybrid_rrf"
    assert shared.dense_score == 0.8
    assert shared.bm25_score == 5.0
    assert shared.rrf_score is not None


def test_hybrid_rrf_merges_duplicate_chunk_ids():
    dense = _FakeDense([_item("shared", 0.9)])
    bm25 = _FakeBM25([_item("shared", 4.0, source="bm25")])
    retriever = HybridRRFRetriever(dense, bm25, fetch_k=1, rrf_k=60)

    rows = retriever.retrieve("query", top_k=10)

    assert [row.chunk_id for row in rows] == ["shared"]
    assert rows[0].dense_score == 0.9
    assert rows[0].bm25_score == 4.0


def test_reranker_final_top_k_limits_final_contexts():
    class FakeReranker:
        def rerank(self, question, items, top_k):
            return list(reversed(items))[:top_k]

    raw, final, warnings, reranker_used = retrieve_top_k_unique_contexts(
        "query",
        _FakeDense([_item("c1", 0.9), _item("c2", 0.8), _item("c3", 0.7), _item("c4", 0.6)]),
        reranker=FakeReranker(),
        top_k=2,
        fetch_k=4,
        max_candidates=4,
    )

    assert len(raw) == 4
    assert [row.chunk_id for row in final] == ["c4", "c3"]
    assert warnings == []
    assert reranker_used is True


def test_dense_old_configs_still_validate_and_build():
    cfg = PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
            "reranker": {"enabled": False},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )

    retriever = build_retriever(cfg.retrieval, _FakeEmbedder(), _FakeIndex(), [_chunk("c1", "alpha")])

    assert retriever.__class__.__name__ == "DenseRetriever"


def test_legacy_placeholder_hybrid_value_is_rejected():
    with pytest.raises(Exception):
        PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "exp", "output_dir": "runs"},
                "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "faiss", "metric": "cosine"},
                "retrieval": {"retriever_type": "hybrid", "top_k": 1, "fetch_k": 1},
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
                "telemetry": {},
                "runtime": {},
            }
        )


class _FakeDense:
    def __init__(self, rows):
        self.rows = rows
        self.last_dense_candidates = []

    def retrieve(self, question, top_k):
        self.last_dense_candidates = self.rows[:top_k]
        return self.last_dense_candidates

    def extract_query_metadata(self, question):
        return None


class _FakeBM25:
    def __init__(self, rows):
        self.rows = rows
        self.last_bm25_candidates = []

    def retrieve(self, question, top_k):
        self.last_bm25_candidates = self.rows[:top_k]
        return self.last_bm25_candidates


class _FakeEmbedder:
    pass


class _FakeIndex:
    pass


def _item(chunk_id: str, score: float, source: str = "dense") -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"{chunk_id}.txt",
        text=chunk_id,
        score=score,
        dense_score=score if source == "dense" else None,
        bm25_score=score if source == "bm25" else None,
        retrieval_source=source,
    )
