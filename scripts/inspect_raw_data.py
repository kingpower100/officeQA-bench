from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MAX_RECORDS = 4
MAX_VALUE_CHARS = 700

FIELD_GROUPS = {
    "question_ids": ("id", "question_id", "qid", "query_id", "qa_id"),
    "context_document_ids": (
        "context_id",
        "context_ids",
        "gold_context_ids",
        "original_context_id",
        "original_context_ids",
        "document_id",
        "doc_id",
        "passage_id",
        "paragraph_id",
    ),
    "question_text": ("question", "query"),
    "context_text": ("context", "cleaned_context", "text", "passage", "paragraph", "document"),
    "table_fields": ("table", "table_text", "table_html", "table_id", "rows", "columns", "cell"),
    "answer_gold_labels": ("answer", "answers", "gold", "label", "target", "derivation", "program_answer", "original_answer"),
    "split_fields": ("split", "subset", "partition", "source_split"),
}


def main() -> None:
    print_usage_note()
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    files = sorted(path for path in RAW_DIR.iterdir() if path.is_file())
    print(f"Inspecting raw data directory: {RAW_DIR}")
    print(f"Files found: {len(files)}")

    for path in files:
        print("\n" + "=" * 100)
        inspect_file(path)


def print_usage_note() -> None:
    print("=" * 100)
    print("Raw Data Audit Notes")
    print("- documents.jsonl should be KB/context only.")
    print("- qa_test/questions_only should be questions only for Pipeline 1.")
    print("- ground_truth_contexts should be evaluation-only.")
    print("- answer fields must be used only by Pipeline 2.")
    print("- Pipeline 1 must not use ground-truth answers or gold context IDs.")
    print("=" * 100)


def inspect_file(path: Path) -> None:
    suffix = path.suffix.lower()
    print(f"File: {path.name}")
    print(f"Extension: {suffix or '(none)'}")
    print(f"Size bytes: {path.stat().st_size}")

    try:
        if suffix == ".jsonl":
            rows = read_jsonl(path)
            print(f"Rows: {len(rows)}")
            print_json_like_summary(rows)
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else [data]
            print(f"Rows: {len(rows) if isinstance(data, list) else '1 JSON document'}")
            print_json_like_summary(rows)
        elif suffix == ".csv":
            rows = read_csv(path)
            print(f"Rows: {len(rows)}")
            print_json_like_summary(rows)
        else:
            print("Rows: n/a")
            preview = path.read_text(encoding="utf-8", errors="replace")[:MAX_VALUE_CHARS]
            print("Preview:")
            print(indent(preview))
    except Exception as exc:
        print(f"ERROR reading {path.name}: {type(exc).__name__}: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                row = {"_line_number": line_number, "_value": row}
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_json_like_summary(records: list[dict[str, Any]]) -> None:
    all_keys = sorted({key for record in records if isinstance(record, dict) for key in record})
    preview_records = records[:MAX_RECORDS]
    print(f"All top-level keys: {', '.join(all_keys) if all_keys else '(none)'}")
    print_field_highlights(all_keys)

    print(f"First {len(preview_records)} record(s):")
    for idx, record in enumerate(preview_records, start=1):
        print(f"\n  Record {idx}:")
        for key, value in record.items():
            flags = classify_key(key)
            marker = f"  <-- {', '.join(flags)}" if flags else ""
            print(f"    {key}: {format_value(value)}{marker}")


def print_field_highlights(keys: Iterable[str]) -> None:
    keys = list(keys)
    print("Likely field groups:")
    for group_name in FIELD_GROUPS:
        matches = [key for key in keys if group_name in classify_key(key)]
        if matches:
            print(f"  - {group_name}: {', '.join(matches)}")
    if not any(classify_key(key) for key in keys):
        print("  - none detected by key name")


def classify_key(key: str) -> list[str]:
    normalized = key.lower().strip()
    matches: list[str] = []
    for group_name, patterns in FIELD_GROUPS.items():
        if any(field_matches(normalized, pattern) for pattern in patterns):
            matches.append(group_name)
    return matches


def field_matches(key: str, pattern: str) -> bool:
    if key == pattern:
        return True
    if pattern == "id":
        return False
    if pattern in {"question", "query", "context", "text", "document"}:
        if key.endswith("_id") or key.endswith("_ids"):
            return False
        return key.endswith(f"_{pattern}") or key.startswith(f"{pattern}_")
    return pattern in key


def format_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", "\\n")
        return repr(truncate(text))
    text = json.dumps(value, ensure_ascii=False, default=str)
    return truncate(text)


def truncate(text: str) -> str:
    if len(text) <= MAX_VALUE_CHARS:
        return text
    return text[:MAX_VALUE_CHARS] + "... [truncated]"


def indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


if __name__ == "__main__":
    main()
