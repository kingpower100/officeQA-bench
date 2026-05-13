from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    text: str
    original_context_id: str | None = None
    metadata: dict = Field(default_factory=dict)
