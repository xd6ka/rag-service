"""Durable job queue: SQLite for state, threads for work, one generation slot.

Why SQLite and not Redis: the workload is one CPU-bound model on one machine.
The queue's job is to make waiting visible and survivable across a restart, not
to distribute work, and a file does that without a second process to operate.
"""
import json
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class QueueFull(RuntimeError):
    pass


class JobStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            self._local.connection = connection
        return connection

    def _init(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    key TEXT UNIQUE,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    result TEXT,
                    error TEXT
                )
                """
            )

    def create(self, question: str, key: str | None = None) -> tuple[dict, bool]:
        job_id = uuid.uuid4().hex[:12]
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO jobs (id, key, question, status, submitted_at) VALUES (?, ?, ?, 'queued', ?)",
                    (job_id, key, question, time.time()),
                )
            return self.get(job_id), False
        except sqlite3.IntegrityError:
            row = connection.execute("SELECT * FROM jobs WHERE key = ?", (key,)).fetchone()
            return dict(row), True

    def set_status(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        fields = ["status = ?"]
        values: list = [status]
        if status == "running":
            fields.append("started_at = ?")
            values.append(time.time())
        if status in {"done", "failed"}:
            fields.append("finished_at = ?")
            values.append(time.time())
        if result is not None:
            fields.append("result = ?")
            values.append(json.dumps(result, ensure_ascii=False))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        with self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", (*values, job_id))

    def get(self, job_id: str) -> dict | None:
        row = self._connect().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def counts(self) -> dict:
        rows = self._connect().execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {row["status"]: row["n"] for row in rows}


class JobQueue:
    def __init__(self, pipeline, store: JobStore, *, workers: int = 1, max_queue: int = 32):
        self.pipeline = pipeline
        self.store = store
        self.max_queue = max_queue
        self._queue: queue.Queue = queue.Queue()
        self._threads = [threading.Thread(target=self._worker, name=f"rag-worker-{i}", daemon=True)
                         for i in range(workers)]
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for thread in self._threads:
            thread.start()
        self._started = True

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def submit(self, question: str, key: str | None = None) -> dict:
        job, duplicate = self.store.create(question, key)
        if duplicate:
            return {"job_id": job["id"], "status": job["status"], "duplicate": True, "position": 0}
        if self.depth >= self.max_queue:
            self.store.set_status(job["id"], "failed", error="queue full")
            raise QueueFull("queue is full")
        self._queue.put(job["id"])
        return {"job_id": job["id"], "status": job["status"], "duplicate": False,
                "position": self.depth}

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.store.get(job_id)
            if job is None or job["status"] != "queued":
                self._queue.task_done()
                continue
            self.store.set_status(job_id, "running")
            queue_ms = (time.time() - job["submitted_at"]) * 1000
            started = time.perf_counter()
            try:
                result = self.pipeline.run(job["question"])
                self.store.set_status(job_id, "done", result=result)
                self.pipeline.metrics.observe(
                    "job", (time.perf_counter() - started) * 1000 + queue_ms, wait_ms=queue_ms
                )
            except Exception as exc:  # noqa: BLE001 - a failed job must not kill the worker
                self.store.set_status(job_id, "failed", error=str(exc))
                self.pipeline.metrics.count("job_errors")
            finally:
                self._queue.task_done()
