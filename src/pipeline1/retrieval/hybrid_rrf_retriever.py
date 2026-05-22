from __future__ import annotations

import time

from src.pipeline1.retrieval.adapters import retrieval_items_to_search_results, search_results_to_retrieval_items, strip_adapter_metadata
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery
from src.pipeline1.retrieval.dense_retriever import DenseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem


class HybridRRFRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BaseRetriever,
        fetch_k: int,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_bm25_candidates: list[RetrievalItem] = []
        self.last_fused_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        results, trace = self.search(SearchQuery(question_id="", query_text=question, top_k=top_k, fetch_k=max(top_k, self.fetch_k)))
        fused = strip_adapter_metadata(search_results_to_retrieval_items(results))
        self.last_fused_candidates = fused
        self.last_retrieval_diagnostics = dict(trace.diagnostics)
        return fused[:top_k]

    def search(self, query: SearchQuery):
        start = time.perf_counter()
        candidate_k = max(query.top_k, query.fetch_k, self.fetch_k)
        dense = self.dense_retriever.retrieve(query.query_text, candidate_k)
        bm25 = self.bm25_retriever.retrieve(query.query_text, candidate_k)
        fused = self._fuse(dense, bm25)
        self.last_dense_candidates = dense
        self.last_bm25_candidates = bm25
        self.last_fused_candidates = fused
        diagnostics = {
            **getattr(self.dense_retriever, "last_retrieval_diagnostics", {}),
            "hybrid_dense_candidates": len(dense),
            "hybrid_bm25_candidates": len(bm25),
            "hybrid_fused_candidates": len(fused),
            "dedupe_policy": DedupePolicy.CHUNK_ID.value,
        }
        results = retrieval_items_to_search_results(fused[:query.top_k])
        return results, RetrievalTrace(
            question_id=query.question_id,
            backend="hybrid_rrf",
            query_latency_ms=(time.perf_counter() - start) * 1000,
            raw_results_count=len(dense) + len(bm25),
            final_results_count=len(results),
            dedupe_policy=DedupePolicy.CHUNK_ID.value,
            filters_applied=query.filters,
            diagnostics=diagnostics,
        )

    def extract_query_metadata(self, question: str):
        return self.dense_retriever.extract_query_metadata(question)

    def _fuse(self, dense: list[RetrievalItem], bm25: list[RetrievalItem]) -> list[RetrievalItem]:
        by_chunk: dict[str, RetrievalItem] = {}
        rrf_scores: dict[str, float] = {}
        dense_scores: dict[str, float | None] = {}
        bm25_scores: dict[str, float | None] = {}
        first_seen: dict[str, int] = {}
        seen_index = 0

        for rank, item in enumerate(dense, start=1):
            key = str(item.chunk_id)
            by_chunk.setdefault(key, item)
            first_seen.setdefault(key, seen_index)
            seen_index += 1
            dense_scores[key] = item.dense_score if item.dense_score is not None else item.score
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(bm25, start=1):
            key = str(item.chunk_id)
            by_chunk.setdefault(key, item)
            first_seen.setdefault(key, seen_index)
            seen_index += 1
            bm25_scores[key] = item.bm25_score if item.bm25_score is not None else item.score
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.bm25_weight * (1.0 / (self.rrf_k + rank))

        fused = []
        for key, item in by_chunk.items():
            score = rrf_scores.get(key, 0.0)
            fused.append(
                item.model_copy(
                    update={
                        "score": score,
                        "dense_score": dense_scores.get(key),
                        "bm25_score": bm25_scores.get(key),
                        "rrf_score": score,
                        "rerank_score": None,
                        "ranking_score_type": "rrf_score",
                        "retrieval_source": "hybrid_rrf",
                    }
                )
            )
        return sorted(fused, key=lambda item: (-item.score, first_seen.get(item.chunk_id, 0)))
