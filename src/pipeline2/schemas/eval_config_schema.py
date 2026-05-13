from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.pipeline2.config_loader import load_eval_config_payload


class EvaluationConfig(BaseModel):
    eval_run_id: str
    output_dir: str = "data/eval/runs/pipeline2"


class InputsConfig(BaseModel):
    rag_outputs: list[str]
    qa_path: str = "data/raw/qa_test.jsonl"
    gold_contexts_path: str = "data/raw/gold_contexts.jsonl"


class RetrievalEvalConfig(BaseModel):
    k: int = Field(default=5, gt=0)


class AnswerQualityConfig(BaseModel):
    enable_numeric_accuracy: bool = True


class OptionalMetricsConfig(BaseModel):
    enable_ragas: bool = False
    enable_llm_judge: bool = False


class RuntimeConfig(BaseModel):
    overwrite: bool = True
    save_csv: bool = True


class EvalConfig(BaseModel):
    evaluation: EvaluationConfig
    inputs: InputsConfig
    retrieval: RetrievalEvalConfig = RetrievalEvalConfig()
    answer_quality: AnswerQualityConfig = AnswerQualityConfig()
    optional_metrics: OptionalMetricsConfig = OptionalMetricsConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        return cls.model_validate(load_eval_config_payload(path))
