import pytest

from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Error, ElasticsearchBM25Retriever
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.retrieval.hybrid_rrf_retriever import HybridRRFRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.retrieval import RetrievalItem


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"context-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"file_name": f"{chunk_id}.txt", "document_id": f"doc-{chunk_id}", "chunk_unit": "test"},
    )


def test_elasticsearch_bm25_config_validates():
    cfg = _cfg()

    assert cfg.retrieval.retriever_type == "hybrid_rrf"
    assert cfg.retrieval.bm25.backend == "elasticsearch"
    assert cfg.retrieval.bm25.index_name == "officeqa_chunks"
    assert cfg.retrieval.bm25.allow_fallback is False


def test_elasticsearch_unavailable_fails_clearly():
    with pytest.raises(ElasticsearchBM25Error, match="unavailable"):
        ElasticsearchBM25Retriever(
            chunks=[_chunk("c1", "alpha")],
            host="http://localhost:9200",
            index_name="test_chunks",
            client=_UnavailableClient(),
        )


def test_factory_does_not_fallback_without_explicit_allowance(monkeypatch):
    cfg = _cfg()

    def fail(*args, **kwargs):
        raise ElasticsearchBM25Error("boom")

    monkeypatch.setattr("src.pipeline1.retrieval.factory.ElasticsearchBM25Retriever", fail)

    with pytest.raises(ElasticsearchBM25Error, match="boom"):
        build_retriever(cfg.retrieval, _FakeEmbedder(), _FakeIndex(), [_chunk("c1", "alpha")])


def test_factory_falls_back_only_when_explicitly_allowed(monkeypatch):
    cfg = _cfg()
    cfg.retrieval.bm25.allow_fallback = True

    def fail(*args, **kwargs):
        raise ElasticsearchBM25Error("boom")

    monkeypatch.setattr("src.pipeline1.retrieval.factory.ElasticsearchBM25Retriever", fail)

    retriever = build_retriever(cfg.retrieval, _FakeEmbedder(), _FakeIndex(), [_chunk("c1", "alpha")])

    assert isinstance(retriever, HybridRRFRetriever)
    assert retriever.bm25_retriever.__class__.__name__ == "BM25Retriever"


def test_elasticsearch_candidate_format_compatible_with_pipeline1_outputs():
    retriever = ElasticsearchBM25Retriever(
        chunks=[_chunk("c1", "treasury bulletin veterans expenditures")],
        host="http://localhost:9200",
        index_name="test_chunks",
        client=_SearchClient(),
    )

    rows = retriever.retrieve("veterans expenditures", top_k=1)

    assert rows[0].chunk_id == "c1"
    assert rows[0].original_context_id == "context-c1"
    assert rows[0].metadata["file_name"] == "c1.txt"
    assert rows[0].metadata["document_id"] == "doc-c1"
    assert rows[0].bm25_score == 12.5
    assert rows[0].retrieval_source == "elasticsearch_bm25"


def test_hybrid_rrf_accepts_es_style_bm25_outputs():
    dense = _ListRetriever([_item("dense_only", 0.9, source="dense"), _item("shared", 0.8, source="dense")])
    sparse = _ListRetriever([_item("es_only", 12.5, source="elasticsearch_bm25"), _item("shared", 7.0, source="elasticsearch_bm25")])
    retriever = HybridRRFRetriever(dense, sparse, fetch_k=2)

    rows = retriever.retrieve("query", top_k=3)

    assert {row.chunk_id for row in rows} == {"dense_only", "es_only", "shared"}
    assert next(row for row in rows if row.chunk_id == "shared").bm25_score == 7.0


class _Indices:
    def __init__(self, exists=False):
        self._exists = exists
        self.created = False

    def exists(self, index):
        return self._exists

    def create(self, index, body):
        self.created = True
        self._exists = True

    def delete(self, index):
        self._exists = False


class _UnavailableClient:
    indices = _Indices()

    def ping(self):
        return False


class _SearchClient:
    def __init__(self):
        self.indices = _Indices(exists=False)
        self.operations = None

    def ping(self):
        return True

    def bulk(self, operations, refresh):
        self.operations = operations

    def search(self, index, size, query):
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "c1",
                        "_score": 12.5,
                        "_source": {
                            "chunk_id": "c1",
                            "context_id": "context-c1",
                            "cleaned_context": "treasury bulletin veterans expenditures",
                            "file_name": "c1.txt",
                            "document_id": "doc-c1",
                            "metadata": {"file_name": "c1.txt", "document_id": "doc-c1", "chunk_unit": "test"},
                        },
                    }
                ]
            }
        }


class _FakeEmbedder:
    pass


class _FakeIndex:
    pass


class _ListRetriever:
    def __init__(self, rows):
        self.rows = rows

    def retrieve(self, question, top_k):
        return self.rows[:top_k]

    def extract_query_metadata(self, question):
        return None


def _item(chunk_id: str, score: float, source: str) -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"{chunk_id}.txt",
        text=chunk_id,
        score=score,
        dense_score=score if source == "dense" else None,
        bm25_score=score if source != "dense" else None,
        retrieval_source=source,
    )


def _cfg() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {
                "retriever_type": "hybrid_rrf",
                "top_k": 5,
                "fetch_k": 50,
                "bm25": {
                    "backend": "elasticsearch",
                    "index_name": "officeqa_chunks",
                    "host": "http://localhost:9200",
                    "rebuild_index": True,
                    "allow_fallback": False,
                    "k1": 1.5,
                    "b": 0.75,
                },
            },
            "reranker": {"enabled": True, "model_name": "fake", "final_top_k": 3},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )
