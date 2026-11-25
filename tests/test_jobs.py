"""Jobs: durable state, idempotency, worker lifecycle, backpressure."""
import time

import pytest

from service.backend import StubBackend
from service.cache import TwoTierCache
from service.jobs import JobQueue, JobStore, QueueFull
from service.pipeline import Pipeline


def _pipeline(delay=0.0):
    return Pipeline(StubBackend(delay=delay), TwoTierCache(quantize=False))


def test_store_create_and_idempotency(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    first, duplicate = store.create("q", key="k1")
    assert duplicate is False and first["status"] == "queued"
    second, duplicate = store.create("q", key="k1")
    assert duplicate is True and second["id"] == first["id"]


def test_status_transitions_and_counts(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job, _ = store.create("q")
    store.set_status(job["id"], "running")
    store.set_status(job["id"], "done", result={"answer": "a"})
    record = store.get(job["id"])
    assert record["status"] == "done"
    assert record["finished_at"] is not None
    assert store.counts() == {"done": 1}


def test_worker_completes_a_job(tmp_path):
    queue = JobQueue(_pipeline(delay=0.01), JobStore(tmp_path / "jobs.sqlite"), workers=1, max_queue=4)
    queue.start()
    submitted = queue.submit("how does chunking work?")
    job_id = submitted["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        job = queue.store.get(job_id)
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.02)
    assert job["status"] == "done", job
    assert "stub answer" in job["result"]


def test_failure_is_recorded_and_the_worker_stays_alive(tmp_path):
    class Boom(StubBackend):
        def answer(self, question):
            if "boom" in question:
                raise RuntimeError("model exploded")
            return super().answer(question)

    pipeline = Pipeline(Boom(), TwoTierCache(quantize=False))
    queue = JobQueue(pipeline, JobStore(tmp_path / "jobs.sqlite"), workers=1, max_queue=4)
    queue.start()
    bad = queue.submit("boom")["job_id"]
    good = queue.submit("fine")["job_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        if queue.store.get(bad)["status"] != "queued" and queue.store.get(good)["status"] == "done":
            break
        time.sleep(0.02)
    assert queue.store.get(bad)["status"] == "failed"
    assert "model exploded" in queue.store.get(bad)["error"]
    assert queue.store.get(good)["status"] == "done"


def test_queue_full_is_rejected(tmp_path):
    queue = JobQueue(_pipeline(), JobStore(tmp_path / "jobs.sqlite"), workers=1, max_queue=0)
    with pytest.raises(QueueFull):
        queue.submit("q")


def test_order_is_fifo(tmp_path):
    queue = JobQueue(_pipeline(delay=0.02), JobStore(tmp_path / "jobs.sqlite"), workers=1, max_queue=8)
    queue.start()
    ids = [queue.submit(f"question {index}")["job_id"] for index in range(3)]
    time.sleep(0.3)
    positions = [queue.store.get(job_id)["started_at"] for job_id in ids]
    assert positions == sorted(positions)
