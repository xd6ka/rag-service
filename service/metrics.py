"""Latency accounting: percentiles per route, plus the queue-versus-compute split.

A p95 that mixes "waited 40 s behind another request" with "the model took 25 s"
tells an operator nothing actionable, so both are recorded separately.
"""
import statistics
import threading
import time
from collections import deque


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; q in [0, 100]. None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 1)


class Metrics:
    def __init__(self, window: int = 500):
        self.window = window
        self._lock = threading.Lock()
        self._latencies: dict[str, deque] = {}
        self._waits: dict[str, deque] = {}
        self._counters: dict[str, int] = {}
        self.started = time.time()

    def observe(self, route: str, ms: float, wait_ms: float = 0.0) -> None:
        with self._lock:
            self._latencies.setdefault(route, deque(maxlen=self.window)).append(ms)
            self._waits.setdefault(route, deque(maxlen=self.window)).append(wait_ms)
            self._counters[route] = self._counters.get(route, 0) + 1

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def snapshot(self) -> dict:
        with self._lock:
            routes = {}
            for route, values in self._latencies.items():
                sample = list(values)
                waits = list(self._waits.get(route, []))
                routes[route] = {
                    "count": len(sample),
                    "p50": percentile(sample, 50),
                    "p90": percentile(sample, 90),
                    "p95": percentile(sample, 95),
                    "p99": percentile(sample, 99),
                    "max": round(max(sample), 1) if sample else None,
                    "mean": round(statistics.mean(sample), 1) if sample else None,
                    "wait_p95": percentile(waits, 95),
                    "wait_mean": round(statistics.mean(waits), 1) if waits else None,
                }
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "routes": routes,
                "counters": dict(self._counters),
            }
