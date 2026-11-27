"""Re-aggregate the mixed-load reports from their stored per-request rows.

The rows are the measurement; the summary is arithmetic over them. Adding
``first_ms``/``min_ms`` should not require a second four-minute run of the model.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.mixed import summarise  # noqa: E402


def main() -> int:
    for name, keys in (
        ("mixed_fast_path.json", ("during_load", "cold")),
        ("mixed_in_slot.json", ("during_load", "cold")),
        ("mixed_stub.json", ("during_load", "cold")),
    ):
        path = ROOT / "bench" / "reports" / name
        report = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            rows = report.get(f"{key}_rows")
            if rows:
                report[key] = summarise(rows)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded = report["during_load"]
        print(f"{name}: first {loaded['first_ms']} ms, p50 {loaded['p50_ms']} ms, max {loaded['max_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
