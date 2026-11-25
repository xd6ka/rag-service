"""API contract: sync answer, cached repeat, queued job, metrics shape."""
import time

from fastapi.testclient import TestClient

from service.api import create_app
from service.backend import StubBackend


def _client(tmp_path):
    app = create_app(StubBackend(delay=0.02), store_path=str(tmp_path / "jobs.sqlite"))
    return TestClient(app)


def test_ask_then_ask_again_hits_the_cache(tmp_path):
    with _client(tmp_path) as client:
        first = client.post("/ask", json={"question": "how does chunking work?"}).json()
        assert first["cached"] is False
        assert first["sources"]
        second = client.post("/ask", json={"question": "How does chunking work?"}).json()
        assert second["cached"] is True and second["tier"] == "exact"
        assert second["timings"]["generate_ms"] == 0.0
        # A hit is resolved before the generation slot is taken, so it never
        # reports queue wait — that is the whole point of the placement.
        assert second["wait_ms"] == 0.0


def test_job_lifecycle(tmp_path):
    with _client(tmp_path) as client:
        submitted = client.post("/jobs", json={"question": "queued question"}).json()
        assert submitted["status"] == "queued"
        job_id = submitted["job_id"]
        deadline = time.time() + 5
        status = "queued"
        while time.time() < deadline:
            body = client.get(f"/jobs/{job_id}").json()
            status = body["status"]
            if status in {"done", "failed"}:
                break
            time.sleep(0.02)
        assert status == "done"
        assert body["result"]["answer"].startswith("stub answer")


def test_unknown_job_is_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/jobs/nope").status_code == 404


def test_metrics_reports_routes_cache_and_queue(tmp_path):
    with _client(tmp_path) as client:
        client.post("/ask", json={"question": "q1"})
        client.post("/ask", json={"question": "q1"})
        body = client.get("/metrics").json()
        assert body["routes"]["routes"]["ask"]["count"] == 2
        assert body["cache"]["exact_hits"] == 1
        assert body["queue"]["depth"] == 0


def test_empty_question_is_rejected(tmp_path):
    with _client(tmp_path) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
