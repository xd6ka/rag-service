# rag-service

A service layer around the RAG pipeline from
[rag-document-assistant](https://github.com/xd6ka/rag-document-assistant): a synchronous
endpoint with an honest rejection path, a durable job queue, a two-tier answer cache,
and latency measured as p50/p95 instead of "it feels fast".

Two things this repo can tell you that a demo cannot:

- **A cached answer used to wait 7.7 seconds behind a generation.** The cache was
  consulted inside the generation slot, so with one worker a repeat question queued
  behind the model. Moving the lookup in front of the slot made the same hit 58 ms.
  Both placements are still in the code (`RAG_CACHE_BEFORE_SLOT`) and both are measured.
- **Adding workers does not add throughput on a CPU-bound model.** At concurrency 4
  throughput stayed at 0.08 rps while p95 latency went from 18.0 s to 58.6 s. That is
  why `/ask` returns 429 with `Retry-After` instead of letting requests pile up
  invisibly.

## Numbers

Same pipeline, same model (`qwen2.5:3b` via project one's local Ollama setup), one
generation worker, answers capped at 120 tokens, warm-up excluded.

**Real backend**, 4 questions × 2 passes (7 cold + 1 cache hit per level):

| concurrency | cold p50 | cold p95 | cached p50 | hit rate | queue wait p95 | wall (8 req) | throughput |
|---|---|---|---|---|---|---|---|
| 1 | 14.6 s | 18.0 s | 17.1 ms | 0.125 | 0 ms | 101.6 s | 0.08 rps |
| 4 | 51.1 s | 58.6 s | 29.5 ms | 0.125 | 41.0 s | 101.0 s | 0.08 rps |

**Stub backend** (deterministic 0 ms work, to isolate the service layer itself):

| concurrency | cold p50 | cold p95 | throughput | errors |
|---|---|---|---|---|
| 1 | 2.4 ms | 3.8 ms | 357 rps | 0 |
| 4 | 8.3 ms | 12.2 ms | 401 rps | 0 |
| 8 | 16.2 ms | 18.6 ms | 440 rps | 0 |

**Cache placement A/B**, exact hit sent 3 s into a cold generation:

| placement | hit latency | hit queue wait | cold latency |
|---|---|---|---|
| cache inside the generation slot | 7741 ms | 7718 ms | 10726 ms |
| cache before the slot (default) | **58 ms** | 0 ms | 10604 ms |

**Semantic cache threshold** (`bench/cache_threshold.py`, 24 paraphrase pairs, 252
unrelated pairs, project one's `all-MiniLM-L6-v2`):

| threshold | paraphrase pairs served | unrelated pairs served | precision | recall |
|---|---|---|---|---|
| 0.60 (chosen) | 3/24 | 0/252 | 1.000 | 0.125 |

int8 packing of cache vectors: max score drift 0.0016, identical hit/miss decisions
at the chosen threshold on all 48 pairs — the quantisation is free here, which is
also why the README reports it as a number instead of a style choice.

## Endpoints

| route | what it does |
|---|---|
| `POST /ask` | answer, or `429` + `Retry-After` when the generation slot is busy past `RAG_WAIT_TIMEOUT` |
| `POST /jobs` | enqueue; returns `job_id`, `position`, and `duplicate: true` for a repeated `idempotency_key` |
| `GET /jobs/{id}` | status, result, age |
| `GET /warm` | load the embedder and read the index without generating — otherwise the first request pays 30–60 s of loading and lands in the p50 |
| `GET /metrics` | per-route count/p50/p90/p95/p99, queue wait separately, cache tiers, queue depth |
| `POST /cache/clear` | start a level from a cold cache |

`/ask` response:

```json
{"answer": "...", "sources": [{"source": "real.pdf", "page": 3, "score": 0.577}],
 "cached": false, "tier": "miss", "wait_ms": 0.0,
 "timings": {"embed_ms": 41.2, "retrieve_ms": 62.8, "generate_ms": 14506.1, "total_ms": 14610.6}}
```

## Design decisions worth defending

- **The pipeline is imported, not copied.** `LocalRagBackend` puts project one on
  `sys.path` and calls its `DocumentStore`, `Retriever` and `answer_question`. A
  service layer that re-implemented retrieval would be measuring a different system
  than the one whose quality was evaluated.
- **SQLite queue, not Redis.** The workload is one CPU-bound model on one machine.
  The queue's job is to make waiting visible and survive a restart; a file does that
  without a second process to operate.
- **429 over invisible queuing.** A 30-second p95 caused by waiting is worse than a
  fast, explicit rejection: the client learns the system is saturated instead of
  learning that the answer is slow.
- **Latency is split.** `metrics.observe(route, total_ms, wait_ms)` records queue wait
  separately from compute, because "waited 41 s behind someone else" and "the model
  took 41 s" require different fixes.
- **A cache hit records which tier served it.** Exact and semantic misses fail
  differently: a wrong semantic hit returns a confident answer to a question nobody
  asked, so the threshold is measured (table above), not chosen.

## The harness bug this repo found

The first load run reported 385–460 ms for a stub that does 0.3 ms of work. The
handler's own `timings.total_ms` said 0.3 ms, `curl` said 3–4 ms, and a pooled
`httpx.Client` said 2.3 ms. The cause was the benchmark creating a new
`httpx.post()` client per request: on this host, connection setup cost 380–680 ms.
`bench/probe_client.py` keeps that comparison, and `bench/load.py` now uses one
pooled client per level. Without this fix every "service latency" number in the
table above would have been a measurement of the test client.

## Limits

- 4 questions × 2 passes is 8 requests per level: one request is 12.5 points of
  hit rate, and the percentiles rest on 7 cold samples. Treat the real-backend
  numbers as an order of magnitude, not a service-level objective.
- One machine, one CPU, one worker, one model. No GPU run, no multi-worker
  comparison, no load from more than 8 concurrent clients.
- The model is Ollama's `qwen2.5:3b` as shipped (Q4_K_M). Model quantisation levels
  were not compared; the only quantisation measured here is int8 packing of cache
  vectors.
- Warm-up (30–60 s) is excluded from percentiles by design, but it is real: a cold
  process cannot answer in 15 s.
- The semantic tier is weak on this embedder (recall 0.125). The cache's usefulness
  in the tables comes from the exact tier; saying otherwise would be inventing a
  result.

## Run it

The service itself needs little; the real backend needs project one's environment
(torch, chromadb, sentence-transformers, its Ollama config).

```bash
# 1) service env: tests and the stub benchmark
python -m venv venv
venv\Scripts\activate                  # source venv/bin/activate elsewhere
pip install -r requirements.txt
pytest -q                              # 27 tests: API contract, cache, queue, pipeline, metrics

# 2) stub load bench, no model involved
python bench\load.py --backend stub --levels 1,4,8 --out bench\reports\load_stub.json

# 3) real backend: run with project one's interpreter, which already has the RAG stack
..\rag-document-assistant\venv\Scripts\python.exe bench\load.py --backend real ^
    --levels 1,4 --questions 4 --repeats 2 --out bench\reports\load_real.json

# 4) cache placement A/B (about 3 minutes)
..\rag-document-assistant\venv\Scripts\python.exe bench\cache_before_slot.py --questions 4 --tokens 40

# 5) semantic threshold sweep (needs the embedder, no generation)
..\rag-document-assistant\venv\Scripts\python.exe bench\cache_threshold.py
```

Start the API with `uvicorn service.api:app --port 8000` (add `RAG_BACKEND=stub` to run
without the model). Env: `RAG_WORKERS`, `RAG_MAX_QUEUE`, `RAG_WAIT_TIMEOUT`,
`RAG_CACHE_*`, `RAG_METRICS_WINDOW`, `RAG_CACHE_BEFORE_SLOT`, `RAG_DB`, `RAG_ROOT`.

Reports are committed under `bench/reports/`: `load_stub.json`, `load_real.json`,
`cache_before_slot.json`, `cache_threshold.json`.
