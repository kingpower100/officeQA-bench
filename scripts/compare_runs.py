from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _discover_summaries(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/summary_by_experiment.csv") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Print rows from Pipeline 2 summary files.")
    parser.add_argument(
        "summary",
        nargs="*",
        default=None,
    )
    args = parser.parse_args()
    summaries = [Path(raw) for raw in args.summary] if args.summary else _discover_summaries(Path("data/eval/runs/pipeline2"))
    for path in summaries:
        if not path.exists():
            print(f"missing: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                print(row)


if __name__ == "__main__":
    main()
