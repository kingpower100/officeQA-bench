from abc import ABC, abstractmethod

from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord


class BaseChunker(ABC):
    @abstractmethod
    def chunk_documents(self, docs: list[DocumentRecord], show_progress: bool = False) -> list[ChunkRecord]:
        raise NotImplementedError
