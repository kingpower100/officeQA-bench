from __future__ import annotations

import json
import logging

from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.schemas.query import QueryRecord


class JsonlReader:
    @staticmethod
    def read_documents(path: str, require_context_id: bool = False) -> list[DocumentRecord]:
        docs: list[DocumentRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                context_id = row.get("context_id") or row.get("original_context_id")
                doc_id = row.get("document_id") or row.get("id") or context_id
                if require_context_id and not context_id:
                    context_id = doc_id
                docs.append(DocumentRecord(
                    document_id=str(doc_id),
                    original_context_id=str(context_id) if context_id is not None else None,
                    text=str(row.get("text") or row.get("context") or ""),
                    metadata={k: v for k, v in row.items() if k not in {"document_id", "id", "text", "context", "context_id", "original_context_id"}},
                ))
        return docs

    @staticmethod
    def iter_queries(path: str, question_id_field: str, question_field: str, logger: logging.Logger | None = None):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    qid = row.get(question_id_field)
                    question = row.get(question_field)
                    if qid is None or question is None:
                        continue
                    yield QueryRecord(question_id=str(qid), question=str(question))
                except Exception as ex:
                    if logger:
                        logger.warning("Skipping malformed query row: %s", ex)
