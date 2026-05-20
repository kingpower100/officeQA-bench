from __future__ import annotations

import warnings

from src.pipeline1.schemas.retrieval import RetrievalItem


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.requested_device = device
        self.model = CrossEncoder(model_name, device=device)
        self.runtime_device = self._resolve_runtime_device()
        self._validate_device_selection()

    def rerank(self, question: str, items: list[RetrievalItem], top_k: int) -> list[RetrievalItem]:
        if not items:
            return []
        scores = self.model.predict([(question, item.text) for item in items])
        scored = [
            item.model_copy(
                update={
                    "score": float(score) + item.metadata_boost,
                    "rerank_score": float(score),
                    "ranking_score_type": "rerank_score_plus_metadata" if item.metadata_boost else "rerank_score",
                    "retrieval_source": item.retrieval_source,
                }
            )
            for item, score in zip(items, scores)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _resolve_runtime_device(self) -> str:
        device = getattr(self.model, "device", None)
        if device is not None:
            return str(device)
        if hasattr(self.model, "model"):
            try:
                parameter = next(self.model.model.parameters())
                return str(parameter.device)
            except Exception:
                pass
        target = getattr(self.model, "_target_device", None)
        if target is not None:
            return str(target)
        return str(self.requested_device)

    def _validate_device_selection(self) -> None:
        requested_cuda = str(self.requested_device).startswith("cuda")
        runtime_cuda = str(self.runtime_device).startswith("cuda")
        if requested_cuda and not runtime_cuda:
            warnings.warn(
                f"CrossEncoder requested device={self.requested_device!r} but runtime device resolved to {self.runtime_device!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
