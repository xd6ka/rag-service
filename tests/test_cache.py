"""Cache behaviour: tiers, normalization, threshold and expiry."""
import numpy as np

from service.cache import TwoTierCache, cosine, normalize_question, quantize_int8


def _vec(*values):
    return list(np.asarray(values, dtype=np.float32))


def test_normalization_makes_the_same_question_one_key():
    assert normalize_question("  WHAT is  RAG?! ") == normalize_question("what is rag")
    assert normalize_question("Что такое RAG?") != ""


def test_exact_hit_after_put():
    cache = TwoTierCache(threshold=0.9)
    cache.put("what is rag", _vec(1, 0, 0), "answer", [{"source": "a.pdf"}])
    payload, tier = cache.get("What is RAG?", _vec(1, 0, 0))
    assert tier == "exact" and payload["answer"] == "answer"
    assert cache.stats.exact_hits == 1


def test_semantic_hit_for_a_close_paraphrase():
    cache = TwoTierCache(threshold=0.8, quantize=False)
    cache.put("how does chunking work", _vec(1, 0.1, 0), "chunking answer", [])
    payload, tier = cache.get("how does text chunking work", _vec(1, 0.15, 0))
    assert tier == "semantic"
    assert payload["semantic_score"] >= 0.8
    assert cache.stats.semantic_hits == 1


def test_distant_question_is_a_miss():
    cache = TwoTierCache(threshold=0.9, quantize=False)
    cache.put("how does chunking work", _vec(1, 0, 0), "a", [])
    _, tier = cache.get("what is the refund policy", _vec(0, 1, 0))
    assert tier == "miss"
    assert cache.stats.misses == 1


def test_int8_cosine_stays_close_to_float_cosine():
    rng = np.random.default_rng(0)
    a = rng.normal(size=384).astype(np.float32)
    b = (a + 0.05 * rng.normal(size=384)).astype(np.float32)
    qa, _ = quantize_int8(a)
    qb, _ = quantize_int8(b)
    # Near-parallel vectors catch the int8 dot-product overflow that random
    # vectors hide: without the float cast both similarities collapse to ~0.
    assert cosine(a, b) > 0.99
    assert abs(cosine(a, b) - cosine(qa, qb)) < 0.01


def test_ttl_expiry_forces_a_miss():
    cache = TwoTierCache(ttl=0.001, threshold=0.99, quantize=False)
    cache.put("q", _vec(1, 0), "a", [])
    import time

    time.sleep(0.01)
    _, tier = cache.get("q", _vec(1, 0))
    assert tier == "miss"
    assert cache.stats.expired == 1


def test_capacity_evicts_the_oldest_entry():
    cache = TwoTierCache(capacity=2, threshold=0.99, quantize=False)
    cache.put("one", _vec(1, 0), "1", [])
    cache.put("two", _vec(0, 1), "2", [])
    cache.put("three", _vec(1, 1), "3", [])
    assert cache.stats.evictions == 1
    assert cache.get("one", _vec(1, 0))[1] == "miss"
    assert cache.get("three", _vec(1, 1))[1] == "exact"


def test_hit_rate_counts_both_tiers():
    cache = TwoTierCache(threshold=0.9, quantize=False)
    cache.put("q", _vec(1, 0), "a", [])
    cache.get("q", _vec(1, 0))
    cache.get("other", _vec(0, 1))
    assert cache.stats.hit_rate == 0.5
    assert cache.stats.as_dict()["size"] == 1
