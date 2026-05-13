from src.pipeline1.orchestrator import dedupe_retrieval_by_original_context_id
from src.pipeline1.schemas.retrieval import RetrievalItem


def test_dedup_by_original_context_id_preserves_alignment():
    items = [
        RetrievalItem(chunk_id="c1_a", original_context_id="ctx1", text="one", score=0.9),
        RetrievalItem(chunk_id="c1_b", original_context_id="ctx1", text="one duplicate", score=0.8),
        RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.7),
        RetrievalItem(chunk_id="c3", original_context_id="ctx3", text="three", score=0.6),
    ]

    deduped = dedupe_retrieval_by_original_context_id(items, top_k=2)

    assert [item.chunk_id for item in deduped] == ["c1_a", "c2"]
    assert [item.original_context_id for item in deduped] == ["ctx1", "ctx2"]
    assert [item.text for item in deduped] == ["one", "two"]
    assert [item.score for item in deduped] == [0.9, 0.7]
