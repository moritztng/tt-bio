#!/usr/bin/env python3
"""Break the 100-step timing run down by token count.

The stage's batch stream is not one size: WeightedPDBDataset crops to token_budget 384, so an
entry smaller than the crop stays smaller. A single median over a mixed stream hides that, and
the number someone actually wants is the cost at the crop.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    r = json.loads(Path(sys.argv[1]).read_text())
    per_step_tokens = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    times = r["all_s"]
    toks = per_step_tokens or r.get("tokens_per_step")
    if toks is None:
        raise SystemExit("need per-step token counts")
    buckets = defaultdict(list)
    for t, n in zip(times, toks):
        buckets[n].append(t)
    out = {
        "gpu": r["gpu"], "precision": r["precision"], "per_rank_batch": r["per_rank_batch"],
        "token_budget": r["token_budget"], "n_timed": len(times),
        "overall_median_s": statistics.median(times),
        "overall_mean_s": statistics.fmean(times),
        "by_tokens": {
            str(n): {
                "n": len(v), "median_s": statistics.median(v), "mean_s": statistics.fmean(v),
                "min_s": min(v), "max_s": max(v),
            }
            for n, v in sorted(buckets.items())
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
