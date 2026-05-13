from abc import ABC, abstractmethod

import numpy as np


class BaseVectorIndex(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int):
        raise NotImplementedError
