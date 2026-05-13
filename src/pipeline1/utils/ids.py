import hashlib


def make_chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    payload = f"{document_id}:{start}:{end}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
