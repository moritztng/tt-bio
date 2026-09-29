#!/usr/bin/env python3
"""Read one footprint.json into the two numbers the guard needs.

RESIDENT is the card at a round boundary: weights, masks, anything alive between rounds, and
the part two interleaved trajectories SHARE. PEAK is the high-water inside the round, sampled
at every Evoformer/extra-MSA block and at every seam. What a SECOND trajectory adds is the
peak minus the shared resident floor, which is what `refuse_if_it_will_not_fit` prices.

Round 1 is the compile round and is dropped: its samples carry the trunk load.
"""
import json
import sys

GB = 2**30

for path in sys.argv[1:]:
    s = json.load(open(path))["samples"]
    bounds = [i for i, x in enumerate(s) if x[1] == "round_boundary"]
    warm = s[bounds[1]:] if len(bounds) > 1 else s
    good = [x for x in warm if x[2] >= 0]
    if not good:
        print(path, "no good samples")
        continue
    total = good[0][3]
    peak = max(x[2] for x in good)
    floor = min(x[2] for x in good)
    res = [x[2] for x in good if x[1] == "round_boundary"]
    lcf = min(x[4] for x in good)
    print(json.dumps({"file": path, "samples": len(good), "total_gb": round(total / GB, 4),
                      "peak_gb": round(peak / GB, 4), "min_gb": round(floor / GB, 4),
                      "round_boundary_gb": [round(x / GB, 4) for x in res],
                      "free_at_peak_gb": round((total - peak) / GB, 4),
                      "min_largest_contig_gb": round(lcf / GB, 4)}))
