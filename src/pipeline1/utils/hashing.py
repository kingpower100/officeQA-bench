import hashlib
import json
from pathlib import Path


def ensure_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash_dict(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
