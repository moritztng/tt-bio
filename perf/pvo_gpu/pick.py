"""Print the arm with the lowest mean warm s/fold at K=1 among the named arms: pick.py <model dir> <arm>...
An arm with any FAILED round, or none warm, is out. Used on the box to re-rank Mol-1's fast kernels on the A100."""
import json, statistics, sys
from pathlib import Path

best = {}
for arm in sys.argv[2:]:
    f = Path(sys.argv[1]) / f"{arm}-k1" / "runs.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
    warm = [r["wall_s"] for r in rows if r.get("kind") == "warm"]
    if warm and not any(r.get("status") == "FAILED" for r in rows):
        best[arm] = statistics.mean(warm)
print(min(best, key=best.get) if best else sys.argv[2])
