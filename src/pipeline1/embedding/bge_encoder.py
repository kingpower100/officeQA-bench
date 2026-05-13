import numpy as np

from src.pipeline1.embedding.base import BaseEmbedder


class BGEEncoder(BaseEmbedder):
    def __init__(self, model_name: str, normalize_embeddings: bool = True, batch_size: int = 32, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def encode_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]
