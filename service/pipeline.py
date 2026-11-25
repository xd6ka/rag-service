"""The one code path both endpoints use: cache, embed, answer, cache again.

Keeping this in a single place is what makes the API comparison honest: the
synchronous endpoint and the queued job do the same work, so a latency difference
between them is the queue, not a different implementation.
"""
import time

from . import config
from .cache import TwoTierCache
from .metrics import Metrics


class Pipeline:
    def __init__(self, backend, cache: TwoTierCache | None = None, metrics: Metrics | None = None):
        self.backend = backend
        self.cache = cache or TwoTierCache(
            capacity=config.CACHE_CAPACITY,
            ttl=config.CACHE_TTL,
            threshold=config.SIM_THRESHOLD,
            quantize=config.CACHE_QUANTIZE,
        )
        self.metrics = metrics or Metrics(config.METRICS_WINDOW)

    def lookup(self, question: str) -> dict:
        """Embed and consult the cache without touching the generation slot."""
        started = time.perf_counter()
        embed_started = time.perf_counter()
        embedding = self.backend.embed(question)
        embed_ms = (time.perf_counter() - embed_started) * 1000
        payload, tier = self.cache.get(question, embedding)
        return {
            "embedding": embedding,
            "payload": payload,
            "tier": tier,
            "embed_ms": round(embed_ms, 1),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def hit(self, found: dict) -> dict:
        """The cached response. Its latency is embed + lookup, not generation."""
        payload = found["payload"]
        total_ms = found["elapsed_ms"]
        self.metrics.count("cache_hits")
        return {
            "answer": payload["answer"],
            "sources": payload["sources"],
            "cached": True,
            "tier": found["tier"],
            "semantic_score": payload.get("semantic_score"),
            "timings": {"embed_ms": found["embed_ms"], "retrieve_ms": 0.0,
                        "generate_ms": 0.0, "total_ms": round(total_ms, 1)},
        }

    def run(self, question: str, found: dict | None = None) -> dict:
        found = found or self.lookup(question)
        if found["payload"] is not None:
            return self.hit(found)

        started = time.perf_counter()
        result = self.backend.answer(question)
        self.cache.put(question, found["embedding"], result["answer"], result["sources"])
        total_ms = found["elapsed_ms"] + (time.perf_counter() - started) * 1000
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "cached": False,
            "tier": "miss",
            "semantic_score": None,
            "timings": {**result.get("timings", {}), "embed_ms": found["embed_ms"],
                        "total_ms": round(total_ms, 1)},
        }
