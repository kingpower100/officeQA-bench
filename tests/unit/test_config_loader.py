from pathlib import Path

from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def test_pipeline_configs_load_current_examples():
    p1 = PipelineConfig.from_yaml("configs/pipeline1/experiments/exp_001_fixed512_bge_qwen25_7b.yaml")
    p2 = EvalConfig.from_yaml("configs/pipeline2/experiments/eval_exp_001_qwen25_7b.yaml")

    assert p1.experiment.experiment_id == "exp_001_fixed512_bge_qwen25_7b"
    assert p1.retrieval.top_k == 5
    assert p2.evaluation.eval_run_id == "eval_exp_001_qwen25_7b"
    assert p2.retrieval.k == 5


def test_pipeline1_base_uses_question_only_and_safe_run_defaults():
    p1 = PipelineConfig.from_yaml("configs/pipeline1/base.yaml")

    assert p1.data.qa_test_path == "data/raw/questions_only.jsonl"
    assert p1.runtime.resume is False
    assert p1.runtime.overwrite is True
