"""The same fusable-pair ranking, for the diffusion path, from the graph capture.

`fusion_pairs.py` ranks the Pairformer block from the buffer-keyed trace. The diffusion path has no
such trace -- only `perf/b2x_difflayer/graph_512_all.json.gz`, a raw `ttnn.graph` node list with no
tt_bio call-site tags. It carries the one thing that matters anyway: which op ALLOCATED each buffer
and which ops consume it.

The rule is imported, not re-derived. `itemize()` out of `perf/b2x_difflayer/itemize.py` is the
counter `b2x-diffusion-layer-bytes` built and the one `perf/b2x-baseline-attrib/roof_table.py`
already reads rather than vendoring a second copy. A single-use intermediate here is a DRAM buffer
with an allocating op and exactly one consuming op.

Two limits of this capture, stated because they bound what the table can claim:

  * The capture is 4 sampling steps, not 200. Per-call figures are per-call; the fold column
    multiplies by the call counts in `perf/bioir_roofline/flops_bytes_512.json` (4800 token DiT
    layers and 1200 atom transformer layers at 200 steps), which is the same configuration.
  * There are no call-site tags, so pairs are named by op code only. The Pairformer table resolves
    to model functions; this one cannot, and a re-capture with tags is what would fix it.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "perf", "b2x_difflayer"))
sys.path.insert(0, HERE)

from itemize import itemize                                                   # noqa: E402
from real_traffic import NO_TRAFFIC                                           # noqa: E402
from fusion_pairs import classify                                             # noqa: E402

# `itemize` records a buffer against a view op that aliases rather than moves. Counting such a
# buffer as a producer->consumer round trip invents traffic that never happened, which is the same
# class of error as the tensor-id byte count that inflated this fold 1.6x. The rule is imported
# from `real_traffic.py` -- the same NO_TRAFFIC set that file charges nothing -- and extended with
# the two ttnn spellings this capture uses for the same thing.
NO_MOVE = {n.replace("ttnn.", "") for n in NO_TRAFFIC} | {"Tensor.__getitem__", "reallocate"}


def moves_bytes(op: str) -> bool:
    return op.replace("ttnn.", "") not in NO_MOVE

# calls/fold at 512 aa, 200 sampling steps -- perf/bioir_roofline/flops_bytes_512.json
CALLS = {"difftx|1x512x768,1x512x768": ("token DiT layer", 4800),
         "difftx|1x224x32x128,1x224x32x128": ("atom transformer layer", 1200)}


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "perf", "b2x_difflayer", "graph_512_all.json.gz")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fusion_pairs_diff.json")
    cap = json.load(gzip.open(src, "rt"))

    out, fold_avail, fold_blocked, fold_total = [], 0.0, 0.0, 0.0
    for call in cap["calls"]:
        sig = call["sig"]
        if sig not in CALLS:
            continue
        label, n_calls = CALLS[sig]
        ops, rows = itemize(call)
        dram_alloc = sum(r["size"] for r in rows if "DRAM" in r["kind"])

        by_pair = defaultdict(lambda: {"n": 0, "removable_b": 0.0})
        n_su, su_b = 0, 0.0
        for r in rows:
            if "DRAM" not in r["kind"] or r["alloc_op"] is None or r["n_consumers"] != 1:
                continue
            p, c = r["alloc_op"], r["consumer_names"][0]
            # a view on either end means no byte crossed DRAM for this pair, so there is nothing
            # for fusion to delete and counting it would invent traffic
            if not moves_bytes(p) or not moves_bytes(c):
                continue
            n_su += 1
            su_b += r["size"]
            k = (p, c, classify(p, c))
            by_pair[k]["n"] += 1
            by_pair[k]["removable_b"] += 2 * r["size"]     # the write and the read
        ranked = sorted(({"producer": k[0], "consumer": k[1], "class": k[2], **v}
                         for k, v in by_pair.items()), key=lambda x: -x["removable_b"])
        avail = sum(x["removable_b"] for x in ranked if x["class"] in ("EPILOGUE", "PROLOGUE"))
        attn = sum(x["removable_b"] for x in ranked if x["class"] == "ATTN-QKV")
        blocked = sum(x["removable_b"] for x in ranked if x["class"] == "REORDER")
        unk = sum(x["removable_b"] for x in ranked if x["class"] == "UNKNOWN")
        rt = 2 * su_b

        print("=" * 100)
        print(f"{label.upper()}  --  {sig}, {n_calls} calls/fold")
        print("=" * 100)
        print(f"  DRAM allocated in the call           {dram_alloc/1e6:9.1f} MB")
        print(f"  single-use intermediates             {n_su:4d} allocations, "
              f"{rt/1e6:.1f} MB of read+write")
        print(f"    fusable as epilogue/prologue       {avail/1e6:9.1f} MB    "
              f"x {n_calls} calls = {avail*n_calls/1e9:7.1f} GB/fold")
        if attn:
            print(f"    qkv projection into attention      {attn/1e6:9.1f} MB    "
                  f"x {n_calls} calls = {attn*n_calls/1e9:7.1f} GB/fold")
        print(f"    blocked by tile reorder            {blocked/1e6:9.1f} MB    "
              f"x {n_calls} calls = {blocked*n_calls/1e9:7.1f} GB/fold")
        print(f"    unclassified                       {unk/1e6:9.1f} MB")
        print()
        print(f"{'removable MB':>12} {'GB/fold':>9} {'n':>4}  {'class':<9} "
              f"{'producer':<30} -> {'consumer':<30}")
        for x in ranked:
            if x["removable_b"] / 1e6 < 0.5:
                continue
            print(f"{x['removable_b']/1e6:>12.2f} {x['removable_b']*n_calls/1e9:>9.2f} "
                  f"{x['n']:>4}  {x['class']:<9} {x['producer'].replace('ttnn.',''):<30} -> "
                  f"{x['consumer'].replace('ttnn.',''):<30}")
        print()
        fold_avail += (avail + attn) * n_calls
        fold_blocked += blocked * n_calls
        fold_total += rt * n_calls
        out.append({"sig": sig, "label": label, "calls_per_fold": n_calls,
                    "dram_alloc_b": dram_alloc, "n_single_use": n_su,
                    "roundtrip_b": rt, "available_b": avail + attn, "blocked_b": blocked,
                    "unknown_b": unk, "pairs": ranked})

    print("=" * 100)
    print(f"DIFFUSION PATH, BOTH LAYER SHAPES, 200 STEPS")
    print(f"  single-use DRAM round trip           {fold_total/1e9:9.1f} GB/fold")
    print(f"    fusable                            {fold_avail/1e9:9.1f} GB/fold")
    print(f"    blocked by tile reorder            {fold_blocked/1e9:9.1f} GB/fold")
    print("  For scale: the whole fold moved 3.405 TB at the 23.841 s baseline "
          "(state/b2x-baseline-attrib.md:212),")
    print("  of which the diffusion path was 1.320 TB. Both are stale at the 17.34 s tip.")
    json.dump({"src": src, "layers": out,
               "fold_roundtrip_b": fold_total, "fold_available_b": fold_avail,
               "fold_blocked_b": fold_blocked}, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
