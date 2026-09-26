#!/usr/bin/env python3
"""Did the triangle-attention route move the round's BYTES, or only its overlap?

`bcx-p10-bytes` is funded off this. At 2409 GB against a 442.3 GB/s roof the device floor is
5.45 s, so a route that cuts seconds without cutting bytes has improved OVERLAP and has not
moved the 10x bar. That has to be said, not inferred from a faster clock.

Reads `perf/bcx_p10_devmap/devmap.py block` blobs, one per route, n=288, K=1, and sums the
operand and result bytes its OpTimer charged, keyed `stack|block|dir|family`. Byte counts are
deterministic, so reps agree exactly and the card the blob came from does not enter the answer.

Per BLOCK is what the instrument measures. The round column multiplies by the block counts a
BindCraft 2 round runs and is a PROJECTION, labelled as one. Bytes project where seconds do
not: a byte count is additive and deterministic, a duration is neither
(memory `op-level-ratios-do-not-transfer-to-the-round`).

    python3 perf/bcx_p10_stack/bytecmp.py perf/bcx_p10_stack/out/bytes
"""
import json
import pathlib
import sys

BLOCKS = {"evo": 48, "extra": 4}      # what a round runs
ROOF = 442.3e9                        # achievable DRAM B/s, the campaign's number


def totals(path):
    """(stack, dir, family) -> read+written bytes a block, and the per-stack total."""
    d = json.load(open(path))
    fam, stack, seen = {}, {}, set()
    for r in d["records"]:
        # one rep per (mode, stack) is enough: bytes do not vary with the clock, and summing
        # every rep would multiply the block by the rep count.
        key = (r["mode"], r["stack"])
        if key in seen or r["mode"] != "sync":
            continue
        seen.add(key)
        for tag, val in list(r["read"].items()) + list(r["written"].items()):
            st, _blk, dr, f = tag.split("|")
            fam[(st, dr, f)] = fam.get((st, dr, f), 0.0) + float(val)
            stack[st] = stack.get(st, 0.0) + float(val)
    return fam, stack


def main(root):
    root = pathlib.Path(root)
    arms = {}
    for name in ("mat", "agtri", "hifi"):
        p = root / f"blk288_stack_{name}.json"
        if p.exists():
            arms[name] = totals(p)
        else:
            print(f"missing {p.name}")
    if "mat" not in arms:
        return 1

    print("one Evoformer block and one extra-MSA block, n=288, K=1, read+written\n")
    print(f"{'arm':<8}{'evo GB/blk':>12}{'extra GB/blk':>14}{'round GB (proj)':>18}"
          f"{'vs mat':>9}{'floor s':>9}")
    proj = {}
    for name in ("mat", "agtri", "hifi"):
        if name not in arms:
            continue
        stack = arms[name][1]
        evo, ext = stack.get("evo", 0) / 1e9, stack.get("extra", 0) / 1e9
        r = evo * BLOCKS["evo"] + ext * BLOCKS["extra"]
        proj[name] = r
        print(f"{name:<8}{evo:>12.4f}{ext:>14.4f}{r:>18.1f}"
              f"{(proj['mat'] / r if r else 0):>9.3f}{r * 1e9 / ROOF:>9.2f}")

    base = arms["mat"][0]
    print("\nper family, GB a block, only the families that moved:")
    print(f"  {'stack':<7}{'dir':<5}{'family':<18}{'mat':>9}{'agtri':>9}{'hifi':>9}"
          f"{'mat/hifi':>10}")
    moved = 0
    for key in sorted(base, key=lambda k: -base[k]):
        row = [arms[n][0].get(key, 0.0) / 1e9 if n in arms else 0.0
               for n in ("mat", "agtri", "hifi")]
        if max(row) - min(row) < 5e-4:
            continue
        moved += 1
        st, dr, f = key
        print(f"  {st:<7}{dr:<5}{f:<18}{row[0]:>9.4f}{row[1]:>9.4f}{row[2]:>9.4f}"
              f"{(row[0] / row[2] if row[2] else 0):>10.3f}")
    if not moved:
        print("  none -- every family is byte-identical across the three routes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "perf/bcx_p10_stack/out/bytes"))
