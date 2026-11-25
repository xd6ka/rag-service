"""Threshold sweep for the semantic cache, on labelled query pairs.

Two things are measured, because they answer different questions:

* does a paraphrase land above the threshold (recall of same-intent pairs);
* does an unrelated question stay below it (precision, i.e. no wrong hit).

A cache serves an answer, not a document, so a false positive is a wrong answer
delivered confidently. The threshold is chosen where false positives are zero
and recall is still useful, and both int8 and float comparisons are reported so
the quantisation cost is visible rather than assumed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.backend import LocalRagBackend  # noqa: E402
from service.cache import cosine, quantize_int8  # noqa: E402

THRESHOLDS = [round(x / 100, 2) for x in range(40, 92, 2)]


def load_pairs(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    same = [(a, b) for group in groups for a in group for b in group if a < b]
    cross = [(a, b) for i, group in enumerate(groups) for other in groups[i + 1:]
             for a in group for b in other]
    return same, cross


def main() -> int:
    backend = LocalRagBackend()
    same, cross = load_pairs(Path(__file__).resolve().parent / "queries.json")
    unique = sorted({q for pair in same for q in pair})
    float_vectors = {q: np.asarray(backend.embed(q), dtype=np.float32) for q in unique}
    int8_vectors = {q: quantize_int8(v)[0] for q, v in float_vectors.items()}

    rows = []
    for threshold in THRESHOLDS:
        for label, pairs, vectors in (
            ("same", same, float_vectors),
            ("cross", cross, float_vectors),
            ("same_int8", same, int8_vectors),
            ("cross_int8", cross, int8_vectors),
        ):
            scores = [cosine(vectors[a], vectors[b]) for a, b in pairs]
            passed = [s for s in scores if s >= threshold]
            rows.append({
                "threshold": threshold,
                "kind": label,
                "pairs": len(scores),
                "above": len(passed),
                "share": round(len(passed) / len(scores), 3),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "mean": round(float(np.mean(scores)), 4),
            })

    # A cache hit answers the question, so a wrong hit is a wrong answer. The
    # operating point is therefore "precision >= 0.9, then as much recall as
    # that allows", and the sweep reports the whole curve instead of one number.
    curve = []
    for threshold in THRESHOLDS:
        cell = {row["kind"]: row for row in rows if row["threshold"] == threshold}
        hits = cell["same"]["above"]
        false_hits = cell["cross"]["above"]
        precision = hits / (hits + false_hits) if hits + false_hits else None
        recall = cell["same"]["share"]
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        curve.append({
            "threshold": threshold,
            "same_above": hits, "cross_above": false_hits,
            "precision": None if precision is None else round(precision, 3),
            "recall": recall,
            "f1": round(f1, 3),
        })
    operating = [point for point in curve if point["precision"] is not None and point["precision"] >= 0.9]
    recommended = max(operating, key=lambda point: (point["f1"], point["recall"])) if operating else None

    summary = {
        "model": "all-MiniLM-L6-v2 (project one embedder)",
        "same_pairs": len(same),
        "cross_pairs": len(cross),
        "rows": rows,
        "curve": curve,
        "recommended_threshold": recommended,
        "float_vs_int8_agreement": {
            "max_score_drift": round(
                max(abs(cosine(float_vectors[a], float_vectors[b]) - cosine(int8_vectors[a], int8_vectors[b]))
                    for a, b in same + cross), 4),
            "decision_agreement_at_recommended": None,
        },
    }
    drift = summary["float_vs_int8_agreement"]
    if recommended:
        threshold = recommended["threshold"]
        agree = 0
        for a, b in same + cross:
            float_hit = cosine(float_vectors[a], float_vectors[b]) >= threshold
            int8_hit = cosine(int8_vectors[a], int8_vectors[b]) >= threshold
            agree += int(float_hit == int8_hit)
        drift["decision_agreement_at_recommended"] = round(agree / (len(same) + len(cross)), 3)

    out = Path(__file__).resolve().parent / "reports" / "cache_threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"same pairs {len(same)}, cross pairs {len(cross)}")
    for point in curve:
        print(f"{point['threshold']:.2f}  same {point['same_above']:2d}/{len(same)}  "
              f"false {point['cross_above']:3d}/{len(cross)}  "
              f"P={point['precision']}  R={point['recall']}")
    print("recommended:", recommended)
    print("max drift:", summary["float_vs_int8_agreement"]["max_score_drift"],
          "agreement:", summary["float_vs_int8_agreement"]["decision_agreement_at_recommended"])
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
