# Demo console

`index.html` is a one-page console served by the API itself at `/demo` (mounted in
`service/api.py`). It is not a mock: it posts to the real `/ask`, polls the real
`/metrics`, and prints the timings and cache tier the service returned. Same origin,
so there is no CORS layer and no second process whose numbers could drift from the
API's.

## What the screenshots show

`console_run.png` — one process, cache cleared before the run, four questions about
the indexed paper each asked twice:

| question | cold | cached | tier |
|---|---|---|---|
| what two components does retrieval-augmented generation combine | 35.4 s | 30 ms | exact |
| what is the difference between RAG-Sequence and RAG-Token | 28.5 s | 29 ms | exact |
| which datasets were used to evaluate RAG | 26.8 s | 27 ms | exact |
| what knowledge source does the retriever use | 24.8 s | 21 ms | exact |
| explain the difference between sequence and token marginalisation | — | 37 ms | semantic |

`/metrics` for that session: `ask` count 9, p50 37 ms, p95 32.7 s, queue wait p95 0 ms,
cache 4 exact + 1 semantic + 4 miss (hit rate 0.556).

`console_answer.png` — the same cached hit with the answer and the pages it was
grounded on: `rag_paper.pdf` p.5, p.2, p.7, p.19, total 35 ms.

## Reproduce it

```bash
# 1) a corpus worth asking about (any substantial PDF works; these screenshots used this one)
curl -L https://arxiv.org/pdf/2005.11401v4 -o ../rag-document-assistant/data/rag_paper.pdf
cd ../rag-document-assistant && python app.py data/rag_paper.pdf     # 117 chunks from 19 pages
cd ../rag-service

# 2) service, warm, cold cache
uvicorn service.api:app --port 8000        # use the RAG project's interpreter: it has torch
# then GET /warm  (pays the 30–70 s model load)  and POST /cache/clear
# open http://127.0.0.1:8000/demo and press "Run 4-question demo"
```

## Two honest notes

- Before the paper was indexed this machine held 5 chunks from two small sample PDFs,
  and the assistant refused every demo question. That refusal was correct — the corpus
  did not contain the answer — and it is also why it made a bad demo. A demo needs a
  corpus that can answer; the evaluation numbers in the main README are unrelated to
  this file.
- The demo session is not the benchmark. Four requests cannot produce a p95; the
  percentiles shown in the console come from the API's own window and are here to prove
  the metrics endpoint is wired to the same code path the service runs, nothing more.
