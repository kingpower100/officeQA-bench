from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.pipeline1.config_loader import load_pipeline_config_payload


class ExperimentConfig(BaseModel):
    experiment_id: str
    random_seed: int = 42
    output_dir: str


class DataConfig(BaseModel):
    documents_path: str
    qa_test_path: str
    question_field: str = "question"
    question_id_field: str = "question_id"
    use_ground_truth_contexts: bool = False
    use_gold_answers: bool = False

    @field_validator("use_ground_truth_contexts", "use_gold_answers")
    @classmethod
    def ensure_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("Pipeline 1 forbids gold answers and ground-truth contexts.")
        return value


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed_token", "fixed_word", "sentence"]
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    tokenizer_name: str = "cl100k_base"


class EmbeddingConfig(BaseModel):
    provider: Literal["sentence_transformers"]
    model_name: str
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"


class IndexConfig(BaseModel):
    type: Literal["faiss"]
    metric: Literal["cosine", "l2"] = "cosine"
    faiss_factory: str = "Flat"
    persist_path: str


class RetrievalConfig(BaseModel):
    retriever_type: Literal["dense"] = "dense"
    top_k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)


class RerankerConfig(BaseModel):
    enabled: bool = False
    model_name: Optional[str] = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("reranker.model_name cannot be blank.")
        return value


class GenerationConfig(BaseModel):
    provider: Literal["ollama"]
    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = Field(default=512, gt=0)
    timeout_s: int = Field(default=90, gt=0)
    system_prompt: str


class PricingConfig(BaseModel):
    input_per_1k_tokens_usd: float = 0.0
    output_per_1k_tokens_usd: float = 0.0


class TelemetryConfig(BaseModel):
    estimate_cost: bool = True
    pricing: PricingConfig = PricingConfig()


class RuntimeConfig(BaseModel):
    num_workers: int = 1
    save_csv: bool = True
    log_level: str = "INFO"
    resume: bool = True
    overwrite: bool = False


class PipelineConfig(BaseModel):
    experiment: ExperimentConfig
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    telemetry: TelemetryConfig
    runtime: RuntimeConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        payload = load_pipeline_config_payload(path, validate_unique_experiment_id=True)
        return cls.model_validate(payload)
