"""Board identity and Tensix grid of one qb1 card, straight out of the perf-page harness.

The p150a / p300c call and the grid are the two facts the board audit could not take from a
planning note. Reuses tt_baseline._card_info (perf_regression.detect_card_type) and
tt_baseline._grid, the same readers every perf-page artifact records, so the probe and a
published artifact name the same part the same way. No fold, no timing: this is provenance,
so it does not need benchlock.
"""
import json, os, sys
sys.path.insert(0, "scripts/gpu_vs_tt")
import tt_baseline as B

out = dict(host=os.uname().nodename,
           card=os.environ.get("TT_VISIBLE_DEVICES"),
           **B._card_info())
out["grid"] = B._grid()
print("PROBE " + json.dumps(out), flush=True)
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
