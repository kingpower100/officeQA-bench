import pytest

from src.pipeline1.generation.ollama_generator import OllamaGenerator


def test_normalize_ollama_base_url_accepts_root_and_api():
    assert OllamaGenerator.normalize_base_url("http://localhost:11434") == "http://localhost:11434"
    assert OllamaGenerator.normalize_base_url("http://localhost:11434/api") == "http://localhost:11434"


def test_normalize_ollama_base_url_rejects_bad_path():
    with pytest.raises(ValueError):
        OllamaGenerator.normalize_base_url("http://localhost:11434/v1")
