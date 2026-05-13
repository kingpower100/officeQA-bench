from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    original_context_id: str | None = None
    text: str
    chunk_start: int
    chunk_end: int
    metadata: dict = Field(default_factory=dict)
