"""Load bench: start the service, warm it, then measure percentile latency.

Cold and cached requests are reported separately. A single p95 that mixes a
25-second model call with a 40-millisecond cache hit is a number nobody can act on.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.metrics import percentile  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(base: str, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/health", timeout=5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("service did not start")


def start_service(port: int, backend: str, wait_timeout: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "RAG_WORKERS": "1",
        "RAG_WAIT_TIMEOUT": wait_timeout,
        "PYTHONUNBUFFERED": "1",
    }
    if backend == "stub":
        env["RAG_BACKEND"] = "stub"
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.api:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_level(base: str, questions: list[str], paraphrases: list[str], concurrency: int, repeats: int) -> dict:
    plan = questions + (paraphrases if repeats >= 2 else []) + (
        questions if repeats >= 3 else []
    )
    results: list[dict] = []
    lock_errors = []
    # One pooled client for the whole level. A fresh httpx.post() per request
    # measured 380-680 ms of connection setup on this host against a 0.3 ms
    # handler: the harness, not the service, was the slow part.
    client = httpx.Client(timeout=900, limits=httpx.Limits(max_connections=concurrency + 4))

    def one(question: str) -> None:
        started = time.perf_counter()
        try:
            response = client.post(f"{base}/ask", json={"question": question})
            client_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                lock_errors.append({"status": response.status_code, "ms": round(client_ms, 1)})
                return
            body = response.json()
            results.append({
                "question": question,
                "client_ms": round(client_ms, 1),
                "server_ms": body["timings"]["total_ms"],
                "wait_ms": body.get("wait_ms", 0.0),
                "cached": body["cached"],
                "tier": body["tier"],
                "generate_ms": body["timings"]["generate_ms"],
            })
        except httpx.HTTPError as exc:
            lock_errors.append({"status": "http_error", "error": str(exc)})

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for question in plan:
                pool.submit(one, question)
    finally:
        client.close()
    wall = time.perf_counter() - started

    cold = [r["client_ms"] for r in results if not r["cached"]]
    cached = [r["client_ms"] for r in results if r["cached"]]
    waits = [r["wait_ms"] for r in results]
    return {
        "concurrency": concurrency,
        "requests": len(plan),
        "completed": len(results),
        "errors": lock_errors[:5],
        "error_count": len(lock_errors),
        "wall_s": round(wall, 1),
        "throughput_rps": round(len(results) / wall, 2) if wall else None,
        "cold": {
            "count": len(cold),
            "p50": percentile(cold, 50), "p95": percentile(cold, 95),
            "max": round(max(cold), 1) if cold else None,
        },
        "cached": {
            "count": len(cached),
            "p50": percentile(cached, 50), "p95": percentile(cached, 95),
            "max": round(max(cached), 1) if cached else None,
        },
        "hit_rate": round(len(cached) / len(results), 3) if results else None,
        "tiers": {tier: sum(1 for r in results if r["tier"] == tier) for tier in {r["tier"] for r in results}},
        "server_wait_mean_ms": round(statistics.mean(waits), 1) if waits else None,
        "server_wait_p95_ms": percentile(waits, 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,4", help="comma-separated concurrency levels")
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backend", default="real", choices=["real", "stub"])
    parser.add_argument("--wait-timeout", default="240")
    parser.add_argument("--out", default=str(ROOT / "bench" / "reports" / "load.json"))
    args = parser.parse_args()

    groups = json.loads((ROOT / "bench" / "queries.json").read_text(encoding="utf-8"))["groups"][: args.questions]
    questions = [group[0] for group in groups]
    # The second pass paraphrases rather than repeating: a repeated string only
    # exercises the exact tier, and the semantic tier is the one with a measured
    # probability of being wrong.
    paraphrases = [group[1] if len(group) > 1 else group[0] for group in groups]
    levels = [int(value) for value in args.levels.split(",") if value.strip()]

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    service = start_service(port, args.backend, args.wait_timeout)
    report = {"backend": args.backend, "questions": questions, "repeats": args.repeats, "levels": []}
    try:
        wait_ready(base)
        warm = httpx.get(f"{base}/warm", timeout=600).json()
        print("warm:", warm)
        report["warm"] = warm
        for level in levels:
            httpx.post(f"{base}/cache/clear", timeout=30)
            time.sleep(0.5)
            row = run_level(base, questions, paraphrases, level, args.repeats)
            row["metrics"] = httpx.get(f"{base}/metrics", timeout=30).json()
            report["levels"].append(row)
            print(f"C={level}: cold p50 {row['cold']['p50']} p95 {row['cold']['p95']} | "
                  f"cached p50 {row['cached']['p50']} | hits {row['hit_rate']} | "
                  f"errors {row['error_count']} | wall {row['wall_s']}s")
    finally:
        service.terminate()
        try:
            service.wait(timeout=15)
        except subprocess.TimeoutExpired:
            service.kill()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
