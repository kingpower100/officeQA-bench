import pytest
import numpy as np

from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex
from src.pipeline1.indexing.factory import build_index
from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Error, ElasticsearchBM25Retriever
from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.retrieval.hybrid_rrf_retriever import HybridRRFRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import IndexConfig
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


def test_elasticsearch_dense_config_validates_type_alias():
    cfg = _cfg(
        index={
            "type": "elasticsearch",
            "host": "http://localhost:9200",
            "index_name": "officeqa_fixed512_bge_small",
            "dense_dim": 384,
            "vector_field": "embedding",
            "text_field": "text",
            "similarity": "cosine",
            "recreate": True,
        },
        retrieval={"type": "elasticsearch_dense", "top_k": 5, "fetch_k": 20},
    )

    assert cfg.index.type == "elasticsearch"
    assert cfg.index.index_name == "officeqa_fixed512_bge_small"
    assert cfg.retrieval.retriever_type == "elasticsearch_dense"
    assert cfg.index.retrieval_mode == "script_score"


def test_elasticsearch_index_creates_mapping_and_bulk_indexes_chunks():
    client = _DenseClient(exists=True)
    index = ElasticsearchIndex(
        host="http://localhost:9200",
        index_name="dense_chunks",
        dense_dim=3,
        recreate=True,
        client=client,
    )
    index.set_chunks([_chunk("c1", "alpha text"), _chunk("c2", "beta text")])

    index.build(np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype="float32"))

    assert client.indices.created_body["mappings"]["properties"]["embedding"]["type"] == "dense_vector"
    assert client.indices.created_body["mappings"]["properties"]["embedding"]["dims"] == 3
    assert client.indices.created_body["mappings"]["properties"]["embedding"]["index"] is True
    assert client.indices.created_body["mappings"]["properties"]["embedding"]["similarity"] == "cosine"
    assert client.indices.created_body["settings"]["number_of_shards"] == 1
    assert client.indices.created_body["settings"]["number_of_replicas"] == 0
    assert client.indices.deleted is True
    assert len(client.operations) == 4
    assert client.operations[1]["chunk_id"] == "c1"
    assert client.operations[1]["original_context_id"] == "context-c1"
    assert client.operations[1]["embedding"] == pytest.approx([0.1, 0.2, 0.3])

    chunk_ids, scores = index.search(np.array([0.1, 0.2, 0.3], dtype="float32"), top_k=1)

    assert chunk_ids == ["c1"]
    assert scores == pytest.approx([0.75])
    assert client.search_calls[-1]["query"] == {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": pytest.approx([0.1, 0.2, 0.3])},
            },
        }
    }


def test_elasticsearch_knn_query_body_is_correct():
    client = _KnnClient(exists=True)
    index = ElasticsearchIndex(
        host="http://localhost:9200",
        index_name="dense_chunks",
        dense_dim=3,
        retrieval_mode="knn",
        num_candidates=42,
        client=client,
    )

    hits = index.search_hits(np.array([0.1, 0.2, 0.3], dtype="float32"), top_k=5)

    assert hits[0]["_id"] == "c1"
    assert client.search_calls[-1]["index"] == "dense_chunks"
    assert client.search_calls[-1]["size"] == 5
    assert client.search_calls[-1]["knn"] == {
        "field": "embedding",
        "query_vector": pytest.approx([0.1, 0.2, 0.3]),
        "k": 5,
        "num_candidates": 42,
    }


def test_elasticsearch_auth_config_is_accepted_by_factory(monkeypatch):
    config = IndexConfig.model_validate(
        {
            "type": "elasticsearch",
            "dense_dim": 3,
            "username": "user",
            "password": "pass",
            "api_key": "key",
            "verify_certs": True,
            "request_timeout": 12,
        }
    )

    def fake_index(**kwargs):
        return kwargs

    monkeypatch.setattr("src.pipeline1.indexing.elasticsearch_index.ElasticsearchIndex", fake_index)

    kwargs = build_index(config)

    assert kwargs["api_key"] == "key"
    assert kwargs["username"] == "user"
    assert kwargs["password"] == "pass"
    assert kwargs["verify_certs"] is True
    assert kwargs["request_timeout"] == 12


@pytest.mark.parametrize("status", ["green", "yellow", "red"])
def test_elasticsearch_health_check_handles_cluster_status(status):
    client = _DenseClient(exists=True, health_status=status)
    index = ElasticsearchIndex(host="http://localhost:9200", index_name="dense_chunks", dense_dim=3, client=client)
    index.set_chunks([_chunk("c1", "alpha")])

    index.build(np.array([[0.1, 0.2, 0.3]], dtype="float32"))

    assert index.last_health["cluster"]["status"] == status
    assert index.last_health["index_exists"] is True


def test_elasticsearch_alias_update_after_successful_build_only():
    client = _DenseClient(exists=False)
    index = ElasticsearchIndex(
        host="http://localhost:9200",
        index_name="dense_chunks",
        index_alias="dense_current",
        index_version="v001",
        dense_dim=3,
        client=client,
    )
    index.set_chunks([_chunk("c1", "alpha")])

    index.build(np.array([[0.1, 0.2, 0.3]], dtype="float32"))

    assert index.index_name == "dense_chunks_v001"
    assert client.indices.alias_updates[-1]["actions"][-1]["add"] == {
        "index": "dense_chunks_v001",
        "alias": "dense_current",
    }

    failing = _FailingBulkClient(exists=False)
    failing_index = ElasticsearchIndex(
        host="http://localhost:9200",
        index_name="dense_chunks",
        index_alias="dense_current",
        index_version="v002",
        dense_dim=3,
        client=failing,
    )
    failing_index.set_chunks([_chunk("c1", "alpha")])

    with pytest.raises(RuntimeError, match="bulk failed"):
        failing_index.build(np.array([[0.1, 0.2, 0.3]], dtype="float32"))
    assert failing.indices.alias_updates == []


def test_elasticsearch_dense_retriever_returns_pipeline1_item_format():
    retriever = ElasticsearchDenseRetriever(
        embedder=_VectorEmbedder(),
        index=_DenseIndex(),
        chunks=[_chunk("c1", "alpha text")],
        top_k=1,
        fetch_k=5,
        metadata_boosting=_NoMetadataBoosting(),
        metadata_filtering=_NoMetadataFiltering(),
    )

    rows = retriever.retrieve("alpha", top_k=1)

    assert rows[0].chunk_id == "c1"
    assert rows[0].original_context_id == "context-c1"
    assert rows[0].text == "alpha text"
    assert rows[0].dense_score == pytest.approx(0.75)
    assert rows[0].score == pytest.approx(0.75)
    assert rows[0].retrieval_source == "elasticsearch_dense"
    assert rows[0].metadata["document_id"] == "doc-c1"


def test_factories_build_elasticsearch_dense_backend(monkeypatch):
    config = IndexConfig.model_validate({"type": "elasticsearch", "dense_dim": 3})

    def fake_index(**kwargs):
        return ("es-index", kwargs)

    monkeypatch.setattr("src.pipeline1.indexing.elasticsearch_index.ElasticsearchIndex", fake_index)

    index, kwargs = build_index(config)

    assert index == "es-index"
    assert kwargs["dense_dim"] == 3

    cfg = _cfg(retrieval={"type": "elasticsearch_dense", "top_k": 1, "fetch_k": 2})
    retriever = build_retriever(cfg.retrieval, _VectorEmbedder(), _DenseIndex(), [_chunk("c1", "alpha")])

    assert isinstance(retriever, ElasticsearchDenseRetriever)


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


class _DenseIndices:
    def __init__(self, exists=False):
        self._exists = exists
        self.created_body = None
        self.deleted = False
        self.refreshed = False
        self.alias_updates = []

    def exists(self, index):
        return self._exists

    def create(self, index, body):
        self.created_body = body
        self._exists = True

    def delete(self, index):
        self.deleted = True
        self._exists = False

    def refresh(self, index):
        self.refreshed = True

    def update_aliases(self, body=None, actions=None):
        self.alias_updates.append(body or {"actions": actions})


class _DenseClient:
    def __init__(self, exists=False, health_status="green"):
        self.indices = _DenseIndices(exists)
        self.operations = []
        self.search_calls = []
        self.cluster = _Cluster(health_status)

    def info(self):
        return {"version": {"number": "8.13.0"}}

    def bulk(self, operations, refresh):
        self.operations.extend(operations)

    def search(self, index, size, query, source):
        self.search_calls.append({"index": index, "size": size, "query": query, "source": source})
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "c1",
                        "_score": 1.75,
                        "_source": {"chunk_id": "c1"},
                    }
                ]
            }
        }

    def count(self, index):
        return {"count": len(self.operations) // 2}


class _Cluster:
    def __init__(self, status):
        self.status = status

    def health(self, index):
        return {"status": self.status, "index": index}


class _KnnClient(_DenseClient):
    def search(self, index, size, knn, source):
        self.search_calls.append({"index": index, "size": size, "knn": knn, "source": source})
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "c1",
                        "_score": 0.75,
                        "_source": {"chunk_id": "c1"},
                    }
                ]
            }
        }


class _FailingBulkClient(_DenseClient):
    def bulk(self, operations, refresh):
        raise RuntimeError("bulk failed")


class _DenseIndex:
    text_field = "text"

    def search_hits(self, query_vec, candidate_k):
        return [
            {
                "_id": "c1",
                "_score": 1.75,
                "_source": {
                    "chunk_id": "c1",
                    "document_id": "doc-c1",
                    "original_context_id": "context-c1",
                    "text": "alpha text",
                    "metadata": {"file_name": "c1.txt", "document_id": "doc-c1", "chunk_unit": "test"},
                },
            }
        ]


class _VectorEmbedder:
    def encode_query(self, question):
        return [0.1, 0.2, 0.3]


class _NoMetadataBoosting:
    enabled = False
    company_weight = 0.3
    year_weight = 0.15
    month_weight = 0.0
    year_month_weight = 0.0
    symbol_weight = 0.2
    file_name_weight = 0.0


class _NoMetadataFiltering:
    enabled = False
    strict = False
    strict_year_match = False
    strict_year_month_match = False


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


def _cfg(index=None, retrieval=None) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": index or {"type": "faiss", "metric": "cosine"},
            "retrieval": retrieval or {
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
