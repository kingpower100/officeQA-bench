from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
