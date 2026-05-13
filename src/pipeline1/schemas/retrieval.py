from pydantic import BaseModel


class RetrievalItem(BaseModel):
    chunk_id: str
    original_context_id: str
    text: str
    score: float
