import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    box = {"elapsed_ms": 0.0}
    try:
        yield box
    finally:
        box["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
