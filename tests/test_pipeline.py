"""Pipeline and cache placement: the paths the API depends on, without a model."""
import time

from service.cache import TwoTierCache
from service.pipeline import Pipeline


class CountingBackend:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.embeds = 0
        self.answers = 0

    def embed(self, text: str) -> list[float]:
        self.embeds += 1
        # Text-dependent, so a different question cannot accidentally look like
        # a paraphrase of the previous one (which would hide a cache miss).
        seed = sum(ord(ch) for ch in text)
        return [1.0, (seed % 7) / 7.0, (seed % 11) / 11.0, 0.5]

    def answer(self, question: str) -> dict:
        self.answers += 1
        if self.delay:
            time.sleep(self.delay)
        return {
            "answer": f"answer to {question}",
            "sources": [{"source": "doc.pdf", "page": 2, "score": 0.8}],
            "timings": {"retrieve_ms": 1.0, "generate_ms": 10.0},
        }


def test_lookup_then_hit_skips_generation():
    backend = CountingBackend()
    pipeline = Pipeline(backend)
    pipeline.run("what is chunking?")
    found = pipeline.lookup("What is chunking?!")
    assert found["payload"] is not None and found["tier"] == "exact"
    before = backend.answers
    hit = pipeline.hit(found)
    assert backend.answers == before
    assert hit["cached"] is True and hit["timings"]["generate_ms"] == 0.0


def test_run_accepts_a_precomputed_lookup_and_does_not_embed_twice():
    backend = CountingBackend()
    pipeline = Pipeline(backend)
    found = pipeline.lookup("brand new question")
    assert found["payload"] is None
    result = pipeline.run("brand new question", found)
    assert result["cached"] is False
    assert backend.embeds == 1 and backend.answers == 1
    assert result["timings"]["embed_ms"] == found["embed_ms"]


def test_second_run_is_served_from_cache():
    backend = CountingBackend()
    pipeline = Pipeline(backend)
    pipeline.run("repeat me")
    result = pipeline.run("repeat me")
    assert result["cached"] is True and backend.answers == 1
    assert pipeline.cache.stats.exact_hits == 1


def test_cache_stats_expose_tiers_and_hit_rate():
    backend = CountingBackend()
    pipeline = Pipeline(backend, cache=TwoTierCache(capacity=4, ttl=60, threshold=0.99))
    pipeline.run("first")
    pipeline.run("first")
    pipeline.run("second")
    stats = pipeline.cache.stats.as_dict()
    assert stats["exact_hits"] == 1 and stats["misses"] == 2
    assert stats["hit_rate"] == round(1 / 3, 3)
