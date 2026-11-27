"""Mixed load: does a cache hit wait behind a generation it did not need?

A single-request A/B (``bench/cache_before_slot.py``) shows the placement costs
40 seconds when the slot happens to be busy. This bench puts real load on the
slot — cold questions that actually run the model — and then asks a question the
cache already knows, several times, from the same client pool. The number that
matters is the cached p50 while the model is saturated: 30 ms means the fast
path is separate, 30 seconds means it is not.

Run it twice, once per placement, because the flag is read at service start:

    python bench/mixed.py --cache-before-slot 1 --out bench/reports/mixed_fast_path.json
    python bench/mixed.py --cache-before-slot 0 --out bench/reports/mixed_in_slot.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.load import free_port, wait_ready  # noqa: E402
from service.metrics import percentile  # noqa: E402


def start_service(port: int, backend: str, cache_before_slot: str, wait_timeout: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "RAG_WORKERS": "1",
        "RAG_WAIT_TIMEOUT": wait_timeout,
        "RAG_CACHE_BEFORE_SLOT": cache_before_slot,
        "PYTHONUNBUFFERED": "1",
    }
    if backend == "stub":
        env["RAG_BACKEND"] = "stub"
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.api:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def ask(client: httpx.Client, base: str, question: str) -> dict:
    started = time.perf_counter()
    try:
        response = client.post(f"{base}/ask", json={"question": question}, timeout=600)
    except httpx.HTTPError as exc:
        return {"question": question, "status": "http_error", "error": str(exc),
                "client_ms": round((time.perf_counter() - started) * 1000, 1)}
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    return {
        "question": question,
        "status": response.status_code,
        "retry_after": response.headers.get("retry-after"),
        "client_ms": round((time.perf_counter() - started) * 1000, 1),
        "wait_ms": body.get("wait_ms"),
        "cached": body.get("cached"),
        "tier": body.get("tier"),
        "server_ms": (body.get("timings") or {}).get("total_ms"),
    }


def summarise(rows: list[dict]) -> dict:
    ok = [row["client_ms"] for row in rows if row["status"] == 200]
    refused = [row for row in rows if row["status"] == 429]
    return {
        "attempts": len(rows),
        "served": len(ok),
        "refused": len(refused),
        # "first" matters as much as the median here: with the cache inside the
        # slot exactly one request pays the whole queue, and every later one is
        # fast because the queue has drained. A p50 of six samples hides that.
        "first_ms": ok[0] if ok else None,
        "min_ms": min(ok) if ok else None,
        "p50_ms": percentile(ok, 50),
        "p95_ms": percentile(ok, 95),
        "max_ms": max(ok) if ok else None,
        "retry_after_values": sorted({row["retry_after"] for row in refused}),
        "server_wait_p95_ms": percentile([row["wait_ms"] for row in rows if row["wait_ms"] is not None] or [], 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-before-slot", default="1", choices=["1", "0"])
    parser.add_argument("--backend", default="real", choices=["real", "stub"])
    parser.add_argument("--cold", type=int, default=3, help="cold questions holding the slot")
    parser.add_argument("--hits", type=int, default=6, help="cached questions fired during the load")
    parser.add_argument("--hit-interval", type=float, default=0.4)
    parser.add_argument("--wait-timeout", default="240")
    parser.add_argument("--out", default=str(ROOT / "bench" / "reports" / "mixed.json"))
    args = parser.parse_args()

    groups = json.loads((ROOT / "bench" / "queries.json").read_text(encoding="utf-8"))["groups"]
    warm_question = groups[0][0]
    cold_questions = [group[0] for group in groups[1: 1 + args.cold]]

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    service = start_service(port, args.backend, args.cache_before_slot, args.wait_timeout)
    report = {
        "backend": args.backend,
        "cache_before_slot": args.cache_before_slot == "1",
        "warm_question": warm_question,
        "cold_questions": cold_questions,
    }
    client = httpx.Client(limits=httpx.Limits(max_connections=args.cold + args.hits + 4))
    try:
        wait_ready(base)
        report["warm"] = httpx.get(f"{base}/warm", timeout=600).json()

        # Prime the cache and measure the fast path with an idle slot.
        report["prime"] = ask(client, base, warm_question)
        idle = [ask(client, base, warm_question) for _ in range(3)]
        report["idle"] = summarise(idle)

        # Saturate the slot with cold questions, then ask the cached one.
        httpx.post(f"{base}/cache/clear", timeout=30)
        prime = ask(client, base, warm_question)      # re-prime after the clear
        report["prime_after_clear"] = prime
        during: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.cold) as pool:
            cold_futures = [pool.submit(ask, client, base, question) for question in cold_questions]
            time.sleep(1.0)                            # let the first generation start
            for _ in range(args.hits):
                during.append(ask(client, base, warm_question))
                time.sleep(args.hit_interval)
            cold_rows = [future.result() for future in cold_futures]
        report["cold"] = summarise(cold_rows)
        report["during_load"] = summarise(during)
        report["during_load_rows"] = during
        report["cold_rows"] = cold_rows
        report["metrics"] = httpx.get(f"{base}/metrics", timeout=30).json()
        report["health"] = httpx.get(f"{base}/health", timeout=30).json()
    finally:
        client.close()
        service.terminate()
        try:
            service.wait(timeout=15)
        except subprocess.TimeoutExpired:
            service.kill()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    placement = "cache ahead of the slot" if report["cache_before_slot"] else "cache inside the slot"
    idle_p50 = report["idle"]["p50_ms"]
    loaded = report["during_load"]
    print(f"{placement}: idle cached p50 {idle_p50} ms | cached during load p50 {loaded['p50_ms']} ms "
          f"(served {loaded['served']}/{loaded['attempts']}) | cold p50 {report['cold']['p50_ms']} ms")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
