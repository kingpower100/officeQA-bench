from abc import ABC, abstractmethod

from src.pipeline1.schemas.retrieval import RetrievalItem


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        raise NotImplementedError
