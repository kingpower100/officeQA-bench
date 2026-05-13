import os

from src.pipeline1.generation.ollama_generator import OllamaGenerator
from src.pipeline1.schemas.config_schema import GenerationConfig


def build_generator(config: GenerationConfig):
    return OllamaGenerator(
        model_name=config.model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", config.base_url),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
