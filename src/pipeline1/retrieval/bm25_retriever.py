from __future__ import annotations

import math
import re
from collections import Counter

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(chunk.text) for chunk in chunks]
        self._doc_lens = [len(tokens) for tokens in self._tokenized]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        self._term_freqs = [Counter(tokens) for tokens in self._tokenized]
        self._idf = self._build_idf()
        self.last_bm25_candidates: list[RetrievalItem] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        query_terms = _tokenize(question)
        if not query_terms or not self.chunks:
            return []
        query_terms = list(dict.fromkeys(query_terms))
        scored = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score(query_terms, idx)
            if score <= 0:
                continue
            scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [
            RetrievalItem(
                chunk_id=chunk.chunk_id,
                original_context_id=chunk.original_context_id or chunk.document_id,
                text=chunk.text,
                score=float(score),
                dense_score=None,
                bm25_score=float(score),
                rrf_score=None,
                rerank_score=None,
                ranking_score_type="bm25_score",
                retrieval_source="bm25",
                chunk_unit=chunk.metadata.get("chunk_unit"),
                metadata=dict(chunk.metadata),
            )
            for score, _, chunk in scored[:top_k]
        ]
        self.last_bm25_candidates = ranked
        return ranked

    def extract_query_metadata(self, question: str):
        from src.pipeline1.retrieval.metadata import extract_query_metadata

        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))

    def _build_idf(self) -> dict[str, float]:
        n_docs = len(self._tokenized)
        doc_freqs: Counter[str] = Counter()
        for tokens in self._tokenized:
            doc_freqs.update(set(tokens))
        return {
            term: math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            for term, df in doc_freqs.items()
        }

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        tf = self._term_freqs[doc_index]
        doc_len = self._doc_lens[doc_index]
        if doc_len == 0 or self._avgdl == 0:
            return 0.0
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score += idf * (freq * (self.k1 + 1.0) / denom)
        return score


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())
