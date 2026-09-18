"""The 0.5175 s transpose class, split into the part that is addressing and the part that is data.

`state/c12-profiled-fold.md` puts `TransposeDeviceOperation` at 0.5175 s of 13.2090 s of in-fold
device time at 512 aa (qb2, 1350 MHz sampled during the fold). Whether any of that is recoverable
the way Anthropic's ending-node kernel recovers it -- swap two token-axis strides in a TMA
descriptor, keep the channel axis innermost, move nothing
(`kernels/trimul/native/pkg/v5/python/trimul_native/kernel.py:341-344`) -- is a question about WHICH
AXES each of our transposes swaps, because TTNN tiles the last two axes of a TILE-layout tensor:

  UNTILED      every moved axis sits outside the tiled pair. Tile blocks are relabelled and no
               element leaves its tile, so this is the form that could be pure addressing.
  TILED_INNER  the moved axes ARE the tiled pair. A real element transpose, done tile-locally by
               the hardware; tt_bio measures the 4-D form at ~0.2 ms (`tenstorrent.py:6178`).
  MIXED        an untiled axis is exchanged with a tiled one, so rows move between tiles. This is
               the row-granular scatter `tenstorrent.py:3784-3792` measures at 1.479 ms to DRAM on
               a 320x320x256 bf16 pair tensor, 70.9 GB/s against 373.3 GB/s for a plain clone of
               the same bytes -- 19 % of the copy roof.

Their mechanism needs the swapped axes to be free, and in a `[N,N,c]` TILE tensor the innermost two
axes are (N, c): one token axis is tiled and the other is not, so their swap is our MIXED, not our
UNTILED. That is the asymmetry this script measures rather than assumes.

Reads the whole-fold transpose census from `capture.py` (every `ttnn.transpose` / `ttnn.permute`
call in one 512 aa fold, with its rank, permutation, shape and bytes) and prints the partition.

Usage:  python3 perf/anthro_zpass/tsplit.py [transposes_512.json.gz] [--class-s 0.5175]
"""
import argparse
import gzip
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ADDRESSING = ("UNTILED",)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("census", nargs="?",
                   default=os.path.join(HERE, "out", "c1", "transposes_512.json.gz"))
    p.add_argument("--class-s", type=float, default=0.5175,
                   help="the measured TransposeDeviceOperation class seconds this partitions")
    p.add_argument("--bar-s", type=float, default=0.15,
                   help="the brief's kill bar on the addressing share")
    a = p.parse_args()

    c = json.load(gzip.open(a.census, "rt"))["calls"]
    n, b = Counter(), Counter()
    for x in c:
        n[x["kind"]] += 1
        b[x["kind"]] += x["bytes"]
    tb = sum(b.values()) or 1

    print("=" * 92)
    print(f"TRANSPOSE CENSUS, one 512 aa fold — {len(c)} ttnn.transpose / ttnn.permute calls")
    print("=" * 92)
    print(f"{'kind':<14}{'calls':>7}{'bytes (GB)':>13}{'byte share':>12}")
    for k in ("UNTILED", "TILED_INNER", "MIXED", "NOOP", "UNREADABLE"):
        print(f"{k:<14}{n[k]:>7}{b[k] / 1e9:>13.3f}{100 * b[k] / tb:>11.2f}%")
    print(f"{'TOTAL':<14}{len(c):>7}{tb / 1e9:>13.3f}")

    addr = sum(n[k] for k in ADDRESSING)
    # The partition is by CALL COUNT, not by a rate: zero calls of a kind is zero seconds of it,
    # whatever the per-call cost. That is the only way to price a share of a measured class
    # without transplanting a rate, which `roof-catalogue-rate-must-match-shipped-kernel` forbids.
    a_s = 0.0 if addr == 0 else float("nan")
    print(f"\nADDRESSING (untiled) share of the {a.class_s:.4f} s class: {a_s:.4f} s "
          f"from {addr} calls")
    print(f"DATA-MOVEMENT (tiled) share:                      "
          f"{a.class_s - (a_s if addr == 0 else 0):.4f} s from {len(c) - addr} calls")
    print(f"bar: {a.bar_s:.2f} s -> "
          f"{'DROP, no kernel worth writing' if a_s == 0 or a_s < a.bar_s else 'survives'}")

    print("\nMIXED by call site (their token-axis swap is this kind, not the addressing kind)")
    m = defaultdict(lambda: [0, 0])
    for x in c:
        if x["kind"] != "MIXED":
            continue
        k = (x["owner"], tuple(x["perm"]), x["shape"])
        m[k][0] += 1
        m[k][1] += x["bytes"]
    for k in sorted(m, key=lambda k: -m[k][1]):
        print(f"  {m[k][0]:>5} calls {m[k][1] / 1e9:>8.3f} GB  {k[1]} {k[2]:<18} {k[0]}")

    print("\nby top-level unit")
    g = defaultdict(lambda: [0, 0])
    for x in c:
        u = x["unit"][0].split("#")[0] if x["unit"] else "outside-units"
        g[(u, x["kind"])][0] += 1
        g[(u, x["kind"])][1] += x["bytes"]
    for k in sorted(g, key=lambda k: -g[k][1]):
        print(f"  {k[0]:<24}{k[1]:<13}{g[k][0]:>6} calls {g[k][1] / 1e9:>8.3f} GB")


if __name__ == "__main__":
    main()
