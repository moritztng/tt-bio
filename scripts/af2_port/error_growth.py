"""How fast does the trunk's error grow per block, and how much of it reaches the filter?

Pass 6 read a per-op ratio of 1.0-1.3x and an end-to-end ratio of 32-82x off the same device
trunk and called the difference a coherent systematic term. It is not: it is a gain. The block
taps in a tap-gate report pin the error at blocks 0, 1, 2, 23, 46 and 47, and the two arms are
separated by the slope of `log(1 - pcc)` against the block index, not by its intercept. At block
0 a faithful bfloat16 torch arm and the ttnn arm carry the SAME error; the ttnn arm's error grows
faster per block, and 48 blocks of compounding plus a structure module and four recycles of
`prev_pos` feedback turn that slope difference into the filter miss.

That makes the per-block growth rate the instrument this port was missing. A per-op ratio is
blind to it by construction -- one op against its own captured input pays the injection once and
never sees the gain -- which is why halving the worst per-op terms with TT_BIO_SOFTMAX_CKC moved
the end-to-end numbers by nothing.

    python3 scripts/af2_port/error_growth.py parity_artifacts/laczc128_b80/*.json
"""
from __future__ import annotations

import json
import math
import sys

#: The block taps every tap-gate report carries. Sparse on purpose: the capture stores six
#: Evoformer blocks, which is four more than a slope needs.
BLOCKS = (0, 1, 2, 23, 46, 47)

#: The amplifiers between the trunk and the filter, all of them host torch in every arm.
DOWNSTREAM = ("evoformer#3/pair", "linear/single_activations#0/out",
              "structure_module#0/final_atom_positions",
              "structure_module#3/final_atom_positions",
              "predicted_lddt_head#3/logits", "predicted_aligned_error_head#3/logits")


def growth(pcc: dict, track: str) -> tuple[float, list[tuple[int, float]]]:
    """Geometric growth rate of `1 - pcc` per Evoformer block, least squares in the log."""
    pts = [(b, 1.0 - pcc[f"evoformer_iteration#{b}/{track}"]) for b in BLOCKS
           if f"evoformer_iteration#{b}/{track}" in pcc and pcc[f"evoformer_iteration#{b}/{track}"] < 1.0]
    if len(pts) < 3:
        return float("nan"), pts
    n = len(pts)
    sx = sum(b for b, _ in pts)
    sy = sum(math.log(e) for _, e in pts)
    sxy = sum(b * math.log(e) for b, e in pts)
    sxx = sum(b * b for b, _ in pts)
    return math.exp((n * sxy - sx * sy) / (n * sxx - sx * sx)), pts


def main(paths: list[str]) -> int:
    print("%-34s %-4s %9s %10s %10s %8s" %
          ("report", "trk", "gain/blk", "1-pcc b0", "1-pcc b47", "b47/b0"))
    curves = {}
    for path in paths:
        report = json.load(open(path))
        if "rows" not in report:
            continue
        pcc = {r["tap"]: r["pcc"] for r in report["rows"] if "pcc" in r}
        label = path.rsplit("/", 1)[-1].removesuffix(".json")
        curves[label] = pcc
        for track in ("pair", "msa"):
            rate, pts = growth(pcc, track)
            print("%-34s %-4s %9.4f %10.2e %10.2e %8.1f" %
                  (label, track, rate, pts[0][1], pts[-1][1], pts[-1][1] / pts[0][1]))
    print()
    print("%-42s %s" % ("downstream tap", "  ".join("%-12s" % k[:12] for k in curves)))
    for tap in DOWNSTREAM:
        cells = []
        for pcc in curves.values():
            cells.append("%-12.2e" % (1.0 - pcc[tap]) if tap in pcc else "%-12s" % "-")
        print("%-42s %s" % (tap, "  ".join(cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
