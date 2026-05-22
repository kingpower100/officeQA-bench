import pytest

from scripts import smoke_elasticsearch_backend as smoke


def test_smoke_script_builds_known_vectors():
    chunks = smoke.build_synthetic_chunks(3)

    assert chunks[0].chunk_id == smoke.EXPECTED_TOP_CHUNK_ID
    assert chunks[0].embedding == [1.0, 0.0, 0.0]
    assert len(chunks) == 4
    assert all(len(chunk.embedding) == 3 for chunk in chunks)


def test_smoke_script_requires_two_dimensions():
    with pytest.raises(ValueError, match="dense-dim"):
        smoke.build_synthetic_chunks(1)


def test_smoke_script_query_shapes_are_valid():
    query_vector = [1.0, 0.0, 0.0]

    script_score = smoke.script_score_search_kwargs("idx", query_vector, size=3)
    knn = smoke.knn_search_kwargs("idx", query_vector, size=3, num_candidates=10)

    assert script_score["index"] == "idx"
    assert script_score["size"] == 3
    assert script_score["query"]["script_score"]["script"]["params"]["query_vector"] == query_vector
    assert "cosineSimilarity" in script_score["query"]["script_score"]["script"]["source"]
    assert knn["index"] == "idx"
    assert knn["size"] == 3
    assert knn["knn"] == {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 3,
        "num_candidates": 10,
    }


def test_smoke_script_parse_args_supports_required_modes():
    args = smoke.parse_args(["--host", "http://es:9200", "--mode", "knn", "--dense-dim", "384", "--keep-index"])

    assert args.host == "http://es:9200"
    assert args.mode == "knn"
    assert args.dense_dim == 384
    assert args.keep_index is True
