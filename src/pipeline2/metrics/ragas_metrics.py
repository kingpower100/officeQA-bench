from __future__ import annotations


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
    except Exception:
        return False
    return True
