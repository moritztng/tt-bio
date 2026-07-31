"""Report the projected opendde ``m_feat`` DRAM footprint across the AbAg-XM panel.

Reads the per-target projection written by ``project_mfeat.py`` and answers the
question the Wormhole campaign turns on: how much of the 164-target benchmark
exceeds a Wormhole chip's DRAM budget, and what MSA depth cap brings it inside.

Footprint model (p1 s24.1, validated to 0.006-0.044% against four measured
failures):  ``m_feat_bytes = depth * pad32(tokens) * c_m(128) * 2``.
"""

from __future__ import annotations

import argparse
import json
import statistics as st

# Wormhole Galaxy chip: 6 DRAM channels x 2.000 GiB = 12.00 GiB.
# Blackhole p150a:      8 DRAM channels x 3.984 GiB = 31.88 GiB.
WORMHOLE_DRAM_GIB = 12.00
BLACKHOLE_DRAM_GIB = 31.88

# Measured device outcomes on Wormhole, used to bracket the empirical ceiling.
MEASURED = {
    "9yio": ("OOM", 1.708),   # allocator refused 1_833_828_352 B
    "9jkr": ("folded", 0.87),
}


def capped_depth(row: dict, cap: int | None) -> int:
    """Effective MSA depth once each chain a3m is truncated to ``cap`` rows.

    The paired block is a separate, already-shallow contribution and is not
    truncated -- capping it would change which chain pairings the model sees,
    a bigger scientific perturbation than trimming the tail of a single-chain
    search.
    """
    per_chain = row["max_chain_depth"] if cap is None else min(row["max_chain_depth"], cap)
    return per_chain + row["paired"]


def mfeat_gib(row: dict, cap: int | None) -> float:
    return capped_depth(row, cap) * row["pad"] * 128 * 2 / 2**30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projection", help="mfeat_panel.json from project_mfeat.py")
    ap.add_argument("--budget", type=float, default=1.00,
                    help="per-fold m_feat budget in GiB to count violations against")
    args = ap.parse_args()

    rows = json.load(open(args.projection))
    rows.sort(key=lambda r: -r["mfeat_gib"])
    uncapped = [r["mfeat_gib"] for r in rows]

    print("=== validation vs measured Wormhole outcomes ===")
    for target, (outcome, expected) in MEASURED.items():
        row = next((r for r in rows if r["target"] == target), None)
        if row is None:
            print("  %-6s not in panel" % target)
            continue
        delta = 100 * (row["mfeat_gib"] - expected) / expected
        print("  %-6s projected %6.3f GiB   measured %s at %.3f GiB   delta %+.2f%%"
              % (target, row["mfeat_gib"], outcome, expected, delta))

    print()
    print("=== uncapped distribution, all %d targets ===" % len(rows))
    print("  min %.3f   median %.3f   mean %.3f   max %.3f GiB"
          % (min(uncapped), st.median(uncapped), st.mean(uncapped), max(uncapped)))
    print("  chip DRAM: Wormhole %.2f GiB   Blackhole %.2f GiB (%.2fx)"
          % (WORMHOLE_DRAM_GIB, BLACKHOLE_DRAM_GIB, BLACKHOLE_DRAM_GIB / WORMHOLE_DRAM_GIB))

    print()
    print("=== heaviest 12 uncapped ===")
    print("  %-8s %5s %5s %8s %7s %11s" % ("target", "tok", "pad", "depth", "paired", "m_feat GiB"))
    for r in rows[:12]:
        print("  %-8s %5d %5d %8d %7d %11.3f"
              % (r["target"], r["tokens"], r["pad"], r["depth"], r["paired"], r["mfeat_gib"]))

    print()
    print("=== targets over a %.2f GiB budget, by MSA depth cap ===" % args.budget)
    print("  %-10s %-14s %10s %10s %10s" % ("cap", "over budget", "max GiB", "median", "n capped"))
    for cap in (None, 16384, 12288, 8192, 6144, 4096, 2048):
        vals = [mfeat_gib(r, cap) for r in rows]
        over = sum(1 for v in vals if v > args.budget)
        touched = sum(1 for r in rows if cap is not None and r["max_chain_depth"] > cap)
        label = "uncapped" if cap is None else str(cap)
        print("  %-10s %3d / %-8d %10.3f %10.3f %10d"
              % (label, over, len(rows), max(vals), st.median(vals), touched))


if __name__ == "__main__":
    main()
