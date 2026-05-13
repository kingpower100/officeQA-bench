from __future__ import annotations


def compute_retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float | None]:
    ranked = _dedupe_preserving_order(str(item) for item in retrieved_ids[:k] if item is not None)
    gold_set = {str(item) for item in gold_ids if item is not None}
    overlap_set = set(ranked) & gold_set
    overlap = len(overlap_set)

    hit = 1.0 if overlap > 0 else 0.0
    recall = None if not gold_set else overlap / len(gold_set)
    precision = overlap / len(ranked) if ranked else 0.0
    reciprocal_rank = 0.0
    for idx, item in enumerate(ranked, start=1):
        if item in gold_set:
            reciprocal_rank = 1.0 / idx
            break

    return {
        f"hit_at_{k}": hit,
        f"recall_at_{k}": recall,
        f"precision_at_{k}": precision,
        f"mrr_at_{k}": reciprocal_rank,
    }


def _dedupe_preserving_order(items) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
