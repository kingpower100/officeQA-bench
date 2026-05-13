import argparse

from src.pipeline1.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 1 - RAG Execution")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
