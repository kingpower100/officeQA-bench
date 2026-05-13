from __future__ import annotations

import argparse

import requests

from src.pipeline1.generation.ollama_generator import OllamaGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ollama availability.")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    base_url = OllamaGenerator.normalize_base_url(args.base_url)
    response = requests.get(f"{base_url}/api/tags", timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
