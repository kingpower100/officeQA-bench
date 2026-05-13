from src.pipeline1.indexing.faiss_index import FaissIndex
from src.pipeline1.schemas.config_schema import IndexConfig


def build_index(config: IndexConfig):
    return FaissIndex(metric=config.metric)
