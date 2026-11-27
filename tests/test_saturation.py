"""Saturation accounting: waiters are visible, refusals are counted.

The bug these tests pin down was an observability one: ``/health`` reported
``queue_depth: 0`` while three requests were blocked inside ``Semaphore.acquire``
for forty seconds, and a 429 never reached ``/metrics`` at all.
"""
import threading
import time

from fastapi.testclient import TestClient

from service import api
from service.backend import StubBackend
from service.slots import GenerationSlot


def test_the_slot_counts_waiters_and_keeps_a_high_water_mark():
    slot = GenerationSlot(capacity=1, timeout=5)
    reached = threading.Event()
    release = threading.Event()

    def holder():
        with slot.hold() as (admitted, _):
            assert admitted
            reached.set()
            release.wait(5)

    def waiter():
        with slot.hold() as (admitted, _):
            assert admitted

    first = threading.Thread(target=holder)
    first.start()
    reached.wait(5)
    second = threading.Thread(target=waiter)
    second.start()
    deadline = time.time() + 3
    while slot.snapshot()["waiting"] < 1 and time.time() < deadline:
        time.sleep(0.01)
    assert slot.snapshot()["waiting"] == 1          # not 0
    assert slot.snapshot()["active"] == 1
    release.set()
    first.join(5)
    second.join(5)
    snapshot = slot.snapshot()
    assert snapshot["waiting"] == 0 and snapshot["active"] == 0
    assert snapshot["max_waiting"] == 1
    assert snapshot["admitted"] == 2 and snapshot["wait_p95_ms"] is not None


def test_the_slot_refuses_after_its_budget_and_says_how_long_to_wait():
    slot = GenerationSlot(capacity=1, timeout=0.05)
    release = threading.Event()

    def holder():
        with slot.hold():
            release.wait(5)

    first = threading.Thread(target=holder)
    first.start()
    time.sleep(0.02)
    started = time.perf_counter()
    with slot.hold() as (admitted, wait_ms):
        assert admitted is False
        assert 40 <= wait_ms < 400                  # the budget, not an arbitrary delay
    release.set()
    first.join(5)
    assert slot.snapshot()["rejected"] == 1


def test_a_rejected_ask_is_429_with_retry_after_and_lands_in_the_counters(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "WAIT_TIMEOUT_S", 0.05)
    app = api.create_app(StubBackend(delay=0.4), store_path=str(tmp_path / "jobs.sqlite"))
    responses: dict[str, object] = {}

    with TestClient(app) as client:
        def ask(name: str):
            responses[name] = client.post("/ask", json={"question": f"question {name}"})

        slow = threading.Thread(target=ask, args=("slow",))
        slow.start()
        time.sleep(0.05)                            # the first request now holds the slot
        refused = client.post("/ask", json={"question": "question refused"})
        slow.join(10)
        assert refused.status_code == 429
        assert refused.headers["Retry-After"] == "1"   # timeout 0.05 -> 1 second, floor of int+1
        assert responses["slow"].status_code == 200
        metrics = client.get("/metrics").json()
        assert metrics["routes"]["counters"]["ask_rejected"] == 1
        assert metrics["routes"]["routes"]["ask"]["count"] == 1   # only the survivor is a latency sample
        assert metrics["saturation"]["rejected"] == 1


def test_health_reports_the_ask_slot_not_just_the_job_queue(tmp_path):
    app = api.create_app(StubBackend(delay=0.01), store_path=str(tmp_path / "jobs.sqlite"))
    with TestClient(app) as client:
        client.post("/ask", json={"question": "warm"})
        body = client.get("/health").json()
        assert body["queue_depth"] == 0             # nothing queued for /jobs
        assert body["ask"]["capacity"] == 1
        assert body["ask"]["admitted"] == 1 and body["ask"]["max_waiting"] == 0
