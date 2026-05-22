from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DedupePolicy(StrEnum):
    NONE = "none"
    CHUNK_ID = "chunk_id"
    DOCUMENT_ID = "document_id"
    ORIGINAL_CONTEXT_ID = "original_context_id"


@dataclass(frozen=True)
class SearchQuery:
    question_id: str
    query_text: str
    query_embedding: list[float] | None = None
    top_k: int = 5
    fetch_k: int = 20
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    original_context_id: str
    text: str
    score: float
    rank: int | None = None
    retrieval_backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalTrace:
    question_id: str
    backend: str
    query_latency_ms: float
    raw_results_count: int
    final_results_count: int
    dedupe_policy: str = DedupePolicy.CHUNK_ID.value
    filters_applied: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaseIndexBackend(ABC):
    backend_name: str = "unknown"

    def prepare(self) -> None:
        return None

    @abstractmethod
    def search(self, query: SearchQuery) -> tuple[list[SearchResult], RetrievalTrace]:
        raise NotImplementedError

    def upsert_chunks(self, chunks, embeddings=None) -> None:
        raise NotImplementedError

    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"status": "unknown", "backend": self.backend_name}


class BaseRetriever(ABC):
    retriever_name: str = "unknown"

    @abstractmethod
    def search(self, query: SearchQuery) -> tuple[list[SearchResult], RetrievalTrace]:
        raise NotImplementedError
