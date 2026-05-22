from __future__ import annotations

from typing import Callable

from src.pipeline1.retrieval.contracts import DedupePolicy, SearchResult


def dedupe_search_results(
    results: list[SearchResult],
    top_k: int,
    policy: DedupePolicy | str = DedupePolicy.CHUNK_ID,
) -> tuple[list[SearchResult], dict]:
    policy_value = _normalize_policy(policy)
    if policy_value == DedupePolicy.NONE:
        selected = results[:top_k]
        return selected, _diagnostics(policy_value, len(results), len(selected), 0)

    key_fn = _key_fn(policy_value)
    seen: set[str] = set()
    selected: list[SearchResult] = []
    collapsed = 0
    for result in results:
        key = key_fn(result)
        if key in seen:
            collapsed += 1
            continue
        seen.add(key)
        selected.append(result)
        if len(selected) >= top_k:
            break
    return selected, _diagnostics(policy_value, len(results), len(selected), collapsed)


def _normalize_policy(policy: DedupePolicy | str) -> DedupePolicy:
    if isinstance(policy, DedupePolicy):
        return policy
    return DedupePolicy(str(policy))


def _key_fn(policy: DedupePolicy) -> Callable[[SearchResult], str]:
    if policy == DedupePolicy.CHUNK_ID:
        return lambda result: str(result.chunk_id)
    if policy == DedupePolicy.DOCUMENT_ID:
        return lambda result: str(result.document_id)
    if policy == DedupePolicy.ORIGINAL_CONTEXT_ID:
        return lambda result: str(result.original_context_id)
    raise ValueError(f"Unsupported dedupe policy: {policy}")


def _diagnostics(policy: DedupePolicy, raw_count: int, final_count: int, collapsed_count: int) -> dict:
    return {
        "dedupe_policy": policy.value,
        "raw_results_count": raw_count,
        "final_results_count": final_count,
        "duplicate_collapse_count": collapsed_count,
    }
