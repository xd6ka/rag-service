"""FastAPI layer: a synchronous endpoint, a queued one, and the numbers.

The synchronous endpoint is capped at ``WORKERS`` concurrent generations. Past
that it returns 429 with a Retry-After instead of letting requests pile up:
a 30-second p95 caused by invisible queuing is worse than an honest rejection.
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .backend import build_backend
from .cache import TwoTierCache
from .jobs import JobQueue, JobStore, QueueFull
from .metrics import Metrics
from .pipeline import Pipeline
from .slots import GenerationSlot

WAIT_TIMEOUT_S = float(os.getenv("RAG_WAIT_TIMEOUT", "20"))

# The one-page console in docs/demo is served by the service itself, so the demo runs
# against the same origin: no CORS, no second process, and no UI that could display
# numbers the API did not produce.
DEMO_DIR = Path(__file__).resolve().parent.parent / "docs" / "demo"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class JobRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    idempotency_key: str | None = Field(default=None, max_length=200)


def create_app(backend=None, *, store_path: str | None = None, start_workers: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        chosen = backend
        if chosen is None:
            kind = os.getenv("RAG_BACKEND", "").strip().lower()
            chosen = build_backend("stub" if kind == "stub" else None)
        pipeline = Pipeline(chosen)
        store = JobStore(store_path or config.DB_PATH)
        queue = JobQueue(pipeline, store, workers=config.WORKERS, max_queue=config.MAX_QUEUE)
        if start_workers:
            queue.start()
        app.state.pipeline = pipeline
        app.state.queue = queue
        app.state.store = store
        app.state.slot = GenerationSlot(config.WORKERS, WAIT_TIMEOUT_S)
        yield

    app = FastAPI(title="rag-service", version="1.0", lifespan=lifespan)
    if DEMO_DIR.exists():
        app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")

    @app.get("/health")
    def health(request: Request) -> dict:
        pipeline = getattr(request.app.state, "pipeline", None)
        return {
            "status": "ok" if pipeline else "starting",
            "backend": type(pipeline.backend).__name__ if pipeline else None,
            "chunks": getattr(pipeline.backend, "chunks", None) if pipeline else None,
            "workers": config.WORKERS,
            "queue_depth": request.app.state.queue.depth if hasattr(request.app.state, "queue") else 0,
            # Waiting on the generation slot is not the same queue as waiting for a
            # job, and reporting 0 for both made a saturated service look idle.
            "ask": request.app.state.slot.snapshot() if hasattr(request.app.state, "slot") else None,
        }

    @app.get("/warm")
    def warm(request: Request) -> dict:
        """Load the embedder and read the index, without generating an answer.

        Without this, the first request of a benchmark pays 100+ seconds of model
        loading and lands in the p50 of a twelve-sample run.
        """
        pipeline: Pipeline = request.app.state.pipeline
        started = time.perf_counter()
        embedding = pipeline.backend.embed("warmup")
        return {
            "warmed": True,
            "dimensions": len(embedding),
            "seconds": round(time.perf_counter() - started, 2),
        }

    @app.post("/ask")
    def ask(payload: AskRequest, request: Request) -> dict:
        pipeline: Pipeline = request.app.state.pipeline
        found = None
        if config.CACHE_BEFORE_SLOT:
            # A cached answer is not generation work: it must not queue behind one.
            found = pipeline.lookup(payload.question)
            if found["payload"] is not None:
                result = pipeline.hit(found)
                result["wait_ms"] = 0.0
                pipeline.metrics.observe("ask", result["timings"]["total_ms"], 0.0)
                return result

        slot = request.app.state.slot
        with slot.hold() as (admitted, wait_ms):
            if not admitted:
                # Rejected requests are counted, not just discarded: a percentile
                # table that only sees the survivors reports a healthy service
                # while it is turning work away.
                pipeline.metrics.count("ask_rejected")
                raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": str(int(slot.timeout) + 1)})
            result = pipeline.run(payload.question, found)
            result["wait_ms"] = wait_ms
            pipeline.metrics.observe("ask", result["timings"]["total_ms"], wait_ms)
            return result

    @app.post("/jobs", status_code=202)
    def submit(payload: JobRequest, request: Request) -> dict:
        try:
            outcome = request.app.state.queue.submit(payload.question, payload.idempotency_key)
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "10"}) from None
        job = outcome
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "position": job["position"],
            "duplicate": job["duplicate"],
        }

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> dict:
        job = request.app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        result = None
        if job["result"]:
            result = json.loads(job["result"])
        timings = {}
        if job["submitted_at"]:
            end = job["finished_at"] or time.time()
            timings["age_ms"] = round((end - job["submitted_at"]) * 1000, 1)
        return {
            "job_id": job["id"],
            "status": job["status"],
            "question": job["question"],
            "result": result,
            "error": job["error"],
            "timings": timings,
        }

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        pipeline: Pipeline = request.app.state.pipeline
        return {
            "routes": pipeline.metrics.snapshot(),
            "cache": pipeline.cache.stats.as_dict(),
            "queue": {"depth": request.app.state.queue.depth, "counts": request.app.state.store.counts()},
            "saturation": request.app.state.slot.snapshot(),
        }

    @app.post("/cache/clear")
    def clear_cache(request: Request) -> dict:
        pipeline: Pipeline = request.app.state.pipeline
        pipeline.cache = TwoTierCache(
            capacity=config.CACHE_CAPACITY,
            ttl=config.CACHE_TTL,
            threshold=config.SIM_THRESHOLD,
            quantize=config.CACHE_QUANTIZE,
        )
        pipeline.metrics.count("cache_cleared")
        return {"cleared": True}

    return app


def app_factory() -> FastAPI:
    return create_app()


app = app_factory()
