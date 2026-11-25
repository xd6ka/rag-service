"""Does a cache hit have to wait for the model? Measure both placements.

One worker, one CPU-bound model. A repeated question arrives while three cold
questions are generating. If the cache is consulted inside the generation slot,
the hit waits behind those generations; if it is consulted before the slot, the
hit is answered immediately. Same pipeline, different order, measured twice.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(port: int, before_slot: bool) -> subprocess.Popen:
    env = {
        **os.environ,
        "RAG_WORKERS": "1",
        "RAG_WAIT_TIMEOUT": "300",
        "RAG_CACHE_BEFORE_SLOT": "1" if before_slot else "0",
        "PYTHONUNBUFFERED": "1",
    }
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.api:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_ready(base: str, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("service did not start")


def measure(before_slot: bool, questions: list[str], token_cap: str) -> dict:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    os.environ["LLM_MAX_TOKENS"] = token_cap
    service = start(port, before_slot)
    try:
        wait_ready(base)
        warm = httpx.get(f"{base}/warm", timeout=600).json()
        client = httpx.Client(timeout=900)

        # Seed the cache with the question that will arrive again under load.
        seeded = client.post(f"{base}/ask", json={"question": questions[0]}).json()

        cold_result: dict = {}

        def cold_call() -> None:
            started = time.perf_counter()
            response = client.post(f"{base}/ask", json={"question": questions[1]}).json()
            cold_result.update({
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "cached": response["cached"],
            })

        # Deliberate ordering: the cold request takes the only generation slot,
        # and the cached request is sent 3 s later, i.e. squarely during it.
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cold_call)
            time.sleep(3)
            hit_started = time.perf_counter()
            hit = client.post(f"{base}/ask", json={"question": questions[0]}).json()
            hit_ms = round((time.perf_counter() - hit_started) * 1000, 1)
            future.result()
        wall = round(time.perf_counter() - wall_start, 1)
        client.close()

        return {
            "cache_before_slot": before_slot,
            "warm_seconds": warm.get("seconds"),
            "seed_ms": seeded["timings"]["total_ms"],
            "hit_ms": hit_ms,
            "hit_tier": hit["tier"],
            "hit_wait_ms": hit.get("wait_ms"),
            "cold_ms": cold_result.get("ms"),
            "cold_cached": cold_result.get("cached"),
            "batch_wall_s": wall,
        }
    finally:
        service.terminate()
        try:
            service.wait(timeout=15)
        except subprocess.TimeoutExpired:
            service.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--tokens", default="40", help="LLM_MAX_TOKENS for the run")
    parser.add_argument("--out", default=str(ROOT / "bench" / "reports" / "cache_before_slot.json"))
    args = parser.parse_args()

    groups = json.loads((ROOT / "bench" / "queries.json").read_text(encoding="utf-8"))["groups"]
    questions = [group[0] for group in groups[: args.questions]]
    report = {"questions": questions, "token_cap": args.tokens, "runs": []}
    for before_slot in (False, True):
        row = measure(before_slot, questions, args.tokens)
        report["runs"].append(row)
        print(f"before_slot={before_slot}: hit {row['hit_ms']} ms ({row['hit_tier']}, "
              f"waited {row['hit_wait_ms']} ms), cold {row['cold_ms']} ms, wall {row['batch_wall_s']} s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
