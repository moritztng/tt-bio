"""Size up the AbAg-XM panel against a Wormhole chip's DRAM, on both axes.

Reads the per-target projection written by ``project_mfeat.py``.

Two axes matter and only one of them is fixable by configuration:

* ``m_feat``, the c_m=128 MSA projection, scales as ``depth * pad32(tokens)``.
  An MSA depth cap shrinks it. Model: ``depth * pad32(tokens) * 128 * 2``.
* the pair representation and every trimul intermediate scale as
  ``pad32(tokens)**2 * c``. **No MSA cap touches this**, and there is no
  precision or trunk-chunking flag in ``predict``'s surface either.

Measured on the JapanFold Galaxy at a fixed MSA cap of 4096, one run per target
on a worker confirmed fresh: 514 tokens folds, 853 and 1095 tokens run out of
DRAM. So the binding limit is the token count, not the MSA depth, and the
``m_feat`` column below is an upper-bound diagnostic rather than the thing that
decides whether a target runs. Report both; rank by tokens when choosing what
is runnable.
"""

from __future__ import annotations

import argparse
import json
import statistics as st

# Wormhole Galaxy chip: 6 DRAM channels x 2.000 GiB = 12.00 GiB.
# Blackhole p150a:      8 DRAM channels x 3.984 GiB = 31.88 GiB.
WORMHOLE_DRAM_GIB = 12.00
BLACKHOLE_DRAM_GIB = 31.88

# Measured Wormhole outcomes. m_feat GiB is the size of the allocation the
# allocator refused, which is NOT the same as the chip being that full -- a
# reused worker refused 0.531 GiB after an earlier failure, so a failed fold
# does not release its device DRAM. Only runs on a confirmed-fresh worker are
# usable as capacity evidence.
MEASURED = {
    "9yio": ("OOM uncapped / folds at cap 4096", 1.708),
    "9jkr": ("folded", 0.87),
}

# Token-axis ladder at a fixed MSA cap of 4096, fresh worker each time.
TOKEN_LADDER = [
    (514, "9yio", "folds"),
    (853, "9q7y", "OOM (1.133 GiB refused, 138.6 s)"),
    (1095, "9j4c", "OOM (1.576 GiB refused)"),
]


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
    print("=== token axis: measured ladder at MSA cap 4096 (fresh worker each) ===")
    for tokens, target, outcome in TOKEN_LADDER:
        print("  %5d tok  %-6s  %s" % (tokens, target, outcome))
    tok = sorted(r["tokens"] for r in rows)
    print("  panel tokens: min %d  median %d  max %d" % (tok[0], st.median(tok), tok[-1]))
    for edge in (514, 728, 853):
        n = sum(1 for t in tok if t <= edge)
        print("  <= %4d tok : %3d / %d  (%.1f%%)" % (edge, n, len(tok), 100 * n / len(tok)))

    print()
    print("=== depth axis: targets over a %.2f GiB m_feat budget, by MSA depth cap ===" % args.budget)
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
