"""Cache: an exact tier for the same string, a semantic tier for a paraphrase.

Both tiers sit in front of the model, so a wrong hit is worse than a miss: it
returns a confident answer to a question nobody asked. That is why the semantic
threshold is a measured number (bench/cache_threshold.py) and why every hit
records which tier served it.
"""
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np

PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_question(text: str) -> str:
    """Case, accents, punctuation and whitespace do not change the question."""
    lowered = unicodedata.normalize("NFKD", (text or "").lower())
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    # Punctuation becomes a space, so runs of whitespace have to collapse too —
    # otherwise "RAG?!" and "RAG" are two different cache keys.
    return " ".join(PUNCT.sub(" ", lowered).split())


def quantize_int8(vector: "np.ndarray") -> tuple["np.ndarray", float]:
    scale = float(np.max(np.abs(vector))) or 1.0
    return np.round(vector / scale * 127).astype(np.int8), scale


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    # Cast before the dot product: on int8 vectors numpy accumulates in int8 and
    # silently overflows, which turned every quantised similarity into ~0.
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denominator)


@dataclass
class Entry:
    key: str
    vector: "np.ndarray"
    answer: str
    sources: list[dict]
    created: float
    hits: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    size: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else None

    def as_dict(self) -> dict:
        return {
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expired": self.expired,
            "size": self.size,
            "hit_rate": self.hit_rate,
        }


class TwoTierCache:
    def __init__(
        self,
        capacity: int = 128,
        ttl: float = 3600,
        threshold: float = 0.92,
        quantize: bool = True,
    ):
        self.capacity = capacity
        self.ttl = ttl
        self.threshold = threshold
        self.quantize = quantize
        self._entries: dict[str, Entry] = {}
        self._order: list[str] = []
        self.stats = CacheStats()
        self._lock = threading.Lock()

    def _pack(self, embedding: list[float]) -> "np.ndarray":
        vector = np.asarray(embedding, dtype=np.float32)
        if self.quantize:
            return quantize_int8(vector)[0]
        return vector

    def _fresh(self, entry: Entry, now: float) -> bool:
        return now - entry.created <= self.ttl

    def get(self, question: str, embedding: list[float]) -> tuple[dict | None, str]:
        """Return (payload, tier); tier is 'exact', 'semantic' or 'miss'."""
        key = normalize_question(question)
        now = time.time()
        with self._lock:
            self._drop_expired(now)
            exact = self._entries.get(key)
            if exact is not None and self._fresh(exact, now):
                exact.hits += 1
                self.stats.exact_hits += 1
                self._touch(key)
                return self._payload(exact), "exact"

            candidate = self._pack(embedding)
            best_key, best_score = None, 0.0
            for other_key, entry in self._entries.items():
                if not self._fresh(entry, now):
                    continue
                score = cosine(candidate, entry.vector)
                if score > best_score:
                    best_key, best_score = other_key, score
            if best_key is not None and best_score >= self.threshold:
                entry = self._entries[best_key]
                entry.hits += 1
                self.stats.semantic_hits += 1
                self._touch(best_key)
                payload = self._payload(entry)
                payload["semantic_score"] = round(best_score, 4)
                return payload, "semantic"
            self.stats.misses += 1
            return None, "miss"

    def put(self, question: str, embedding: list[float], answer: str, sources: list[dict]) -> None:
        key = normalize_question(question)
        with self._lock:
            if key not in self._entries and len(self._entries) >= self.capacity:
                oldest = self._order.pop(0) if self._order else None
                if oldest and oldest in self._entries:
                    del self._entries[oldest]
                    self.stats.evictions += 1
            self._entries[key] = Entry(
                key=key,
                vector=self._pack(embedding),
                answer=answer,
                sources=sources,
                created=time.time(),
            )
            self._touch(key)
            self.stats.size = len(self._entries)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _drop_expired(self, now: float) -> None:
        for key in [k for k, e in self._entries.items() if not self._fresh(e, now)]:
            del self._entries[key]
            if key in self._order:
                self._order.remove(key)
            self.stats.expired += 1
        self.stats.size = len(self._entries)

    @staticmethod
    def _payload(entry: Entry) -> dict:
        return {"answer": entry.answer, "sources": entry.sources}
