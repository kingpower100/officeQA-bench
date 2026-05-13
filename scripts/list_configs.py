from __future__ import annotations

from pathlib import Path


def main() -> None:
    for path in sorted(Path("configs").rglob("*.yaml")):
        print(path.as_posix())


if __name__ == "__main__":
    main()
