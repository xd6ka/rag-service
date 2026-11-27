"""The generation slot: a semaphore that knows how many callers are waiting on it.

``threading.BoundedSemaphore`` enforces the concurrency limit but reports nothing.
``/health`` therefore said ``queue_depth: 0`` while three requests sat in
``acquire()`` for forty seconds, which is the worst possible answer to "is this
service busy". This wrapper keeps the counters honest: waiters, high-water mark,
rejections, and the wait each admitted request actually paid.
"""
import threading
import time
from contextlib import contextmanager


class GenerationSlot:
    """Admit up to ``capacity`` concurrent generations; refuse after ``timeout`` seconds."""

    def __init__(self, capacity: int, timeout: float):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.timeout = timeout
        self._sem = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0
        self.rejected = 0
        self.max_waiting = 0
        self.waited_ms: list[float] = []
        self._window = 500

    @contextmanager
    def hold(self):
        """Yield ``(admitted, wait_ms)``; ``admitted=False`` means the budget ran out.

        The non-blocking attempt comes first so that ``waiting`` means "blocked in
        the queue", not "called the endpoint". A high-water mark that counts every
        caller is a traffic counter, not a saturation signal.
        """
        if self._sem.acquire(blocking=False):
            admitted, wait_ms = True, 0.0
            with self._lock:
                self.active += 1
        else:
            with self._lock:
                self.waiting += 1
                self.max_waiting = max(self.max_waiting, self.waiting)
            started = time.perf_counter()
            admitted = self._sem.acquire(timeout=self.timeout)
            wait_ms = (time.perf_counter() - started) * 1000
            with self._lock:
                self.waiting -= 1
                if not admitted:
                    self.rejected += 1
                else:
                    self.active += 1
        if admitted:
            with self._lock:
                self.waited_ms.append(round(wait_ms, 1))
                del self.waited_ms[:-self._window]
        try:
            yield admitted, round(wait_ms, 1)
        finally:
            if admitted:
                with self._lock:
                    self.active -= 1
                self._sem.release()

    def snapshot(self) -> dict:
        from .metrics import percentile

        with self._lock:
            waits = list(self.waited_ms)
            return {
                "capacity": self.capacity,
                "timeout_s": self.timeout,
                "active": self.active,
                "waiting": self.waiting,
                "max_waiting": self.max_waiting,
                "rejected": self.rejected,
                "admitted": len(waits),
                "wait_p50_ms": percentile(waits, 50),
                "wait_p95_ms": percentile(waits, 95),
                "wait_max_ms": max(waits) if waits else None,
            }
