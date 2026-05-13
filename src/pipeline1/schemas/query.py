from pydantic import BaseModel


class QueryRecord(BaseModel):
    question_id: str
    question: str
