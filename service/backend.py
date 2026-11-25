"""Backends: the real RAG pipeline from project one, and a deterministic stub.

The stub exists so the queue, cache and API can be tested without loading torch:
those tests are about the service layer, and a test that needs a 90-second model
load is a test nobody runs.
"""
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def answer(self, question: str) -> dict: ...


def _hash_embedding(text: str, dims: int = 64) -> list[float]:
    """Stable pseudo-embedding: same text, same vector, no model involved."""
    vector = [0.0] * dims
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:2], "big") % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class StubBackend:
    """Fast stand-in used by tests and by the API contract check."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def embed(self, text: str) -> list[float]:
        return _hash_embedding(text)

    def answer(self, question: str) -> dict:
        if self.delay:
            time.sleep(self.delay)
        return {
            "answer": f"stub answer for: {question.strip()}",
            "sources": [{"source": "stub.pdf", "page": 1, "score": 0.5}],
            "timings": {"retrieve_ms": 0.0, "generate_ms": round(self.delay * 1000, 1)},
        }


class LocalRagBackend:
    """Imports project one's retrieval and generation and serves them unchanged."""

    def __init__(self, rag_root: Path | str | None = None):
        from . import config

        root = Path(rag_root or config.RAG_ROOT)
        if not root.exists():
            raise RuntimeError(f"RAG project not found at {root}; set RAG_ROOT")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.generation import answer_question  # noqa: PLC0415
        from src.retrieval import Retriever  # noqa: PLC0415
        from src.store import DocumentStore  # noqa: PLC0415
        from src.embeddings import embed_query  # noqa: PLC0415

        self._answer_question = answer_question
        self._embed_query = embed_query
        self._store = DocumentStore()
        if self._store.count() == 0:
            raise RuntimeError("The vector collection is empty: index a PDF in the RAG project first")
        self._retriever = Retriever(self._store)
        self.chunks = self._store.count()

    def embed(self, text: str) -> list[float]:
        vector = self._embed_query(text)
        return [float(v) for v in vector]

    def answer(self, question: str) -> dict:
        started = time.perf_counter()
        contexts = self._retriever.retrieve(question)
        retrieved_ms = (time.perf_counter() - started) * 1000
        if not contexts:
            return {
                "answer": "",
                "sources": [],
                "timings": {"retrieve_ms": round(retrieved_ms, 1), "generate_ms": 0.0},
            }
        started = time.perf_counter()
        text = self._answer_question(question, contexts)
        generated_ms = (time.perf_counter() - started) * 1000
        return {
            "answer": (text or "").strip(),
            "sources": [
                {"source": chunk.source, "page": chunk.page, "score": round(float(chunk.score), 3)}
                for chunk in contexts
            ],
            "timings": {"retrieve_ms": round(retrieved_ms, 1), "generate_ms": round(generated_ms, 1)},
        }


def build_backend(kind: str | None = None):
    from . import config

    kind = (kind or "").lower()
    if kind == "stub":
        return StubBackend()
    return LocalRagBackend(config.RAG_ROOT)
