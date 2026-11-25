"""Configuration for the service layer. Everything is env-overridable."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The RAG pipeline being served is the one from the first project. It is imported
# rather than copied: a service layer that re-implements retrieval is measuring
# a different system than the one the eval scores.
RAG_ROOT = Path(os.getenv("RAG_ROOT", str(PROJECT_ROOT.parent / "rag-document-assistant")))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(RAG_ROOT / "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# One generation worker by default: the bottleneck is a CPU-bound 3B model, and
# more threads do not make a CPU faster — they only move the wait into the queue.
WORKERS = int(os.getenv("RAG_WORKERS", "1"))
MAX_QUEUE = int(os.getenv("RAG_MAX_QUEUE", "32"))

# Cache: an exact tier for repeated strings and a semantic tier for paraphrase.
CACHE_CAPACITY = int(os.getenv("RAG_CACHE_CAPACITY", "128"))
CACHE_TTL = float(os.getenv("RAG_CACHE_TTL", "3600"))
# The semantic threshold is not a taste decision: bench/cache_threshold.py sweeps
# it on labelled paraphrase pairs, and 0.60 is the lowest point that still served
# zero wrong hits to unrelated questions (recall 0.125 — this embedder is a weak
# paraphrase detector, and the README says so instead of hiding it behind 0.92).
SIM_THRESHOLD = float(os.getenv("RAG_CACHE_SIM", "0.60"))
# Compare cached embeddings in int8 instead of float32. The comparison is
# scale-invariant, so the quantised cosine is computable without dequantising.
CACHE_QUANTIZE = os.getenv("RAG_CACHE_QUANTIZE", "1") == "1"

# Percentiles are computed over a bounded window, not the whole process lifetime,
# so a long-running service does not report the p95 of last week's traffic.
METRICS_WINDOW = int(os.getenv("RAG_METRICS_WINDOW", "500"))

DB_PATH = os.getenv("RAG_DB", str(PROJECT_ROOT / "data" / "jobs.sqlite"))

# Whether the cache is consulted before the generation slot is taken. Off is the
# naive placement (cache inside the critical section) and is kept only so the
# benchmark can measure what it costs: with one worker, a cached answer used to
# wait behind a 40-second generation.
CACHE_BEFORE_SLOT = os.getenv("RAG_CACHE_BEFORE_SLOT", "1") == "1"
