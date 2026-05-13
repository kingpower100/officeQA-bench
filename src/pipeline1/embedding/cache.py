from pathlib import Path

import numpy as np


class EmbeddingCache:
    @staticmethod
    def load(path: Path):
        if not path.exists():
            return None
        return np.load(path)

    @staticmethod
    def save(path: Path, embeddings, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        path.with_suffix(path.suffix + ".meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
