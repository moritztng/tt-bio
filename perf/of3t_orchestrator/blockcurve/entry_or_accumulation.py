#!/usr/bin/env python3
"""Does the trunk's excess ACCUMULATE through the backward, or is it established at ENTRY?

`of3t-cotcoh` is dispatched to walk the coherent fraction back from block 47 and its
pre-registration is two-sided on exactly this. The answer is partly already paid for:
`of3t-modelframe`'s ATTRIBUTION.json enumerates twelve blocks with our rel_l2 AND upstream's own
rel_l2 at the same block, which is enough to compare the two accumulation rates.

The backward ENTERS at block 47 and descends to block 0, so "accumulating through the backward"
means growing from 47 toward 0.

This reads a committed artifact and computes nothing on a device. It is a PRIOR for the walk,
not a substitute: the twelve blocks are 79.55 % of the trunk's error mass, not all 48, so the
unenumerated 36 can still move it.
"""
import json, math, pathlib, platform, re, statistics

root = pathlib.Path(__file__).resolve().parents[3]
d = json.loads((root / "perf/of3t_modelframe/ATTRIBUTION.json").read_text())

rows = []
for e in d["by_block"]:
    m = re.search(r"(\d+)", e["group"])
    if m:
        rows.append((int(m.group(1)), e))
rows.sort()

def fit(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    return sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)

idx = [i for i, _ in rows]
ours = [e["rel_l2"] for _, e in rows]
them = [e["upstreams_own_rel_l2_here"] for _, e in rows]
ratio = [e["over_upstream"] for _, e in rows]

# growth through the DESCENT, i.e. from high index to low: fit against (47 - index)
depth = [47 - i for i in idx]
s_ours = fit(depth, [math.log2(y) for y in ours])
s_them = fit(depth, [math.log2(y) for y in them])
s_ratio = fit(depth, [math.log2(y) for y in ratio])

lo = [e for i, e in rows if i <= 11]      # deep in the descent (backward ends here)
hi = [e for i, e in rows if i >= 39]      # where the backward enters

out = {
 "instrument": "perf/of3t_orchestrator/blockcurve/entry_or_accumulation.py",
 "host": platform.node(),
 "source": "perf/of3t_modelframe/ATTRIBUTION.json, 12 of 48 blocks = 79.55 % of trunk error mass",
 "orientation": "the backward ENTERS at block 47 and descends to 0; 'depth' is 47 - block index",
 "per_block": [{"block": i, "ours_rel_l2": e["rel_l2"],
                "upstream_own_rel_l2": e["upstreams_own_rel_l2_here"],
                "over_upstream": e["over_upstream"],
                "pct_trunk_error_mass": e["pct_of_trunk_error_mass"],
                "pct_trunk_reference_mass": e["pct_of_trunk_reference_mass"]} for i, e in rows],
 "slope_log2_per_block_of_descent": {
     "ours": s_ours, "upstream_own": s_them, "our_excess_over_upstream": s_ratio},
 "endpoints": {
     "ours_at_47": rows[-1][1]["rel_l2"], "ours_at_4": [e for i, e in rows if i == 4][0]["rel_l2"],
     "upstream_at_47": rows[-1][1]["upstreams_own_rel_l2_here"],
     "upstream_at_4": [e for i, e in rows if i == 4][0]["upstreams_own_rel_l2_here"]},
 "mean_over_upstream": {"entry_blocks_39_to_47": statistics.mean(e["over_upstream"] for e in hi),
                        "deep_blocks_0_to_11": statistics.mean(e["over_upstream"] for e in lo)},
 "reading": (
   "BOTH sides accumulate through the descent and UPSTREAM ACCUMULATES FASTER. Ours grows "
   "0.9603 -> 2.8089 from block 47 to block 4, a factor 2.93; upstream's own grows 0.2308 -> "
   "1.6706, a factor 7.24. In log2 per block of descent, ours is %+0.4f and upstream's %+0.4f, "
   "so our EXCESS over upstream SHRINKS as the backward descends, at %+0.4f per block. Mean "
   "over_upstream is %.4f at the entry blocks 39-47 and %.4f at the deep blocks 0-11."
   % (s_ours, s_them, s_ratio,
      statistics.mean(e["over_upstream"] for e in hi),
      statistics.mean(e["over_upstream"] for e in lo))),
 "what_it_argues": (
   "Our excess is largest where the backward ENTERS and attenuates as it descends, so the defect "
   "looks like an ENTRY condition rather than something compounding per block. That is a prior "
   "AGAINST any mechanism whose signature is per-block compounding -- including D240, the "
   "fan-out-keyed cotangent dtype -- as the explanation for the excess ACROSS the stack. It does "
   "not clear D240 at the entry itself."),
 "caveats": (
   "Twelve of 48 blocks, chosen by mass, so this is not a uniform sample and the 36 unenumerated "
   "blocks can move it. Block 11 is an outlier at 4.6214 with worst_rel_l2 18.307. And the ratio "
   "is high at 46/47 substantially because UPSTREAM IS VERY ACCURATE THERE (0.1499, 0.2308), not "
   "only because we are bad -- which is exactly why both absolute curves are reported beside it. "
   "A PRIOR for the walk, not a replacement: of3t-cotcoh should measure all 48."),
}
p = pathlib.Path(__file__).with_name("ENTRY_OR_ACCUMULATION.json")
p.write_text(json.dumps(out, indent=1) + "\n")
print("blk   ours    upstream   over_upstream")
for i, e in rows:
    print("%3d  %.4f   %.4f     %.4f" % (i, e["rel_l2"], e["upstreams_own_rel_l2_here"],
                                         e["over_upstream"]))
print("\n" + out["reading"])
print("\n" + out["what_it_argues"])
print("\nwrote", p)
