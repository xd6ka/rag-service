"""Live demo against the real backend: cold vs cache hit, the durable queue, /metrics.

Run with the RAG project's interpreter (it has the RAG stack) while the API is on:

    uvicorn service.api:app --port 8000
    python bench\\demo_live.py

Every number printed here comes from a running process; the transcript is written to
bench/reports/demo_live.json so the demo can be checked after the fact.
"""
import argparse
import json
import time
from pathlib import Path

import httpx

QUESTIONS = [
    "what two components does retrieval-augmented generation combine",
    "what is the difference between RAG-Sequence and RAG-Token",
    "which datasets were used to evaluate RAG",
    "what knowledge source does the retriever use",
]
PARAPHRASE = "explain the difference between sequence and token marginalisation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--question", action="append", default=None, help="override with your own question(s)")
    parser.add_argument("--out", default="bench/reports/demo_live.json")
    args = parser.parse_args()
    if args.question:
        questions = args.question[: args.questions]
    else:
        questions = QUESTIONS[: args.questions]

    rows: list[dict] = []
    with httpx.Client(base_url=args.api, timeout=300) as client:
        rows.append({"step": "warm", "detail": client.get("/warm").json()})
        rows.append({"step": "cache_clear", "detail": client.post("/cache/clear").json()})

        for question in questions:
            cold = client.post("/ask", json={"question": question}).json()
            rows.append({"step": "ask_cold", "question": question, "detail": cold})
            print(f"cold  {cold['timings']['total_ms']:>9.1f} ms  tier={cold['tier']:<6} {question}")
            hit = client.post("/ask", json={"question": question}).json()
            rows.append({"step": "ask_repeat", "question": question, "detail": hit})
            print(f"hit   {hit['timings']['total_ms']:>9.1f} ms  tier={hit['tier']:<6} {question}")

        para = client.post("/ask", json={"question": PARAPHRASE}).json()
        rows.append({"step": "ask_paraphrase", "question": PARAPHRASE, "detail": para})
        print(f"para  {para['timings']['total_ms']:>9.1f} ms  tier={para['tier']}")

        j1 = client.post("/jobs", json={"question": "how is the vector store persisted"}).json()
        j2 = client.post("/jobs", json={"question": "how are scanned pages handled"}).json()
        dup = client.post(
            "/jobs",
            json={"question": "how is the vector store persisted", "idempotency_key": j1.get("idempotency_key", "")},
        ).json()
        rows.append({"step": "jobs_enqueue", "detail": [j1, j2]})
        rows.append({"step": "jobs_duplicate", "detail": dup})
        print(f"jobs  {j1['job_id']} pos={j1.get('position')} | duplicate={dup.get('duplicate')}")

        started = time.perf_counter()
        final: dict = {}
        while time.perf_counter() - started < 180:
            final = client.get(f"/jobs/{j1['job_id']}").json()
            if final.get("status") in {"done", "failed", "error"}:
                break
            time.sleep(1.0)
        waited = round(time.perf_counter() - started, 1)
        rows.append({"step": "jobs_poll", "detail": final, "waited_s": waited})
        print(f"poll  status={final.get('status')} waited={waited}s")

        metrics = client.get("/metrics").json()
        rows.append({"step": "metrics", "detail": metrics})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {out}")
    print(json.dumps(metrics, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
