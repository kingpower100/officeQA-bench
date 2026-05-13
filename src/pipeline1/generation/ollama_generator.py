import logging
from requests import Timeout
from urllib.parse import urlparse

import requests

from src.pipeline1.generation.base import BaseGenerator, GenerationResult
from src.pipeline1.generation.token_counter import count_tokens


class OllamaGenerator(BaseGenerator):
    logger = logging.getLogger("pipeline1")

    def __init__(self, model_name: str, base_url: str, temperature: float, max_tokens: int, timeout_s: int) -> None:
        self.model_name = model_name
        self.base_url = self.normalize_base_url(base_url)
        self.generate_url = f"{self.base_url}/api/generate"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        raw = base_url.strip()
        if not raw:
            raise ValueError("Ollama base_url cannot be empty.")
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Ollama base_url must start with http:// or https://, got: {base_url!r}")
        if not parsed.netloc:
            raise ValueError(f"Ollama base_url must include host and port, got: {base_url!r}")
        path = parsed.path.rstrip("/")
        if path in ("", "/api"):
            return f"{parsed.scheme}://{parsed.netloc}"
        raise ValueError(f"Ollama base_url must point to host root or /api only, got path '{parsed.path}'")

    def generate(self, prompt: str) -> GenerationResult:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        try:
            response = requests.post(self.generate_url, json=payload, timeout=(10, self.timeout_s))
        except Timeout as ex:
            raise TimeoutError(f"Ollama generation timed out after {self.timeout_s}s") from ex
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        return GenerationResult(
            answer=answer,
            input_tokens=int(data.get("prompt_eval_count") or count_tokens(prompt)),
            output_tokens=int(data.get("eval_count") or count_tokens(answer)),
        )
