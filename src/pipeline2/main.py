import argparse

from src.pipeline2.orchestrator import EvaluationOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 2 - Offline Evaluation")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config")
    args = parser.parse_args()
    EvaluationOrchestrator().run(args.config)


if __name__ == "__main__":
    main()
