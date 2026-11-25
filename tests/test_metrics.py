"""Percentiles and the queue-versus-compute split."""
from service.metrics import Metrics, percentile


def test_percentile_endpoints_and_interpolation():
    assert percentile([], 95) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.5
    # Linear interpolation, not "the nearest sample": 95 % of 1..100 sits between
    # 95 and 96. Rounding to one decimal keeps the number readable without
    # pretending the estimate is exact.
    assert abs(percentile(values, 95) - 95.05) < 0.1
    assert percentile(values, 100) == 100.0


def test_p95_is_not_the_mean_when_the_tail_is_slow():
    metrics = Metrics(window=50)
    for _ in range(19):
        metrics.observe("ask", 100.0)
    metrics.observe("ask", 5000.0)
    snapshot = metrics.snapshot()["routes"]["ask"]
    assert snapshot["p50"] < 200
    # With one slow request in twenty, p95 interpolates between the 18th and 19th
    # samples: the tail shows up in the mean and the max, not in p95.
    assert snapshot["p95"] > 300
    assert snapshot["max"] == 5000.0
    assert snapshot["mean"] > snapshot["p50"]


def test_window_bounds_memory():
    metrics = Metrics(window=10)
    for index in range(50):
        metrics.observe("job", float(index))
    snapshot = metrics.snapshot()["routes"]["job"]
    assert snapshot["count"] == 10
    assert snapshot["max"] == 49.0


def test_wait_is_tracked_separately_from_total():
    metrics = Metrics()
    metrics.observe("ask", 30000.0, wait_ms=20000.0)
    snapshot = metrics.snapshot()["routes"]["ask"]
    assert snapshot["wait_p95"] == 20000.0
