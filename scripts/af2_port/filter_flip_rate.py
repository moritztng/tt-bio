"""How often would the device arm's miss change a PXDesign accept/reject?

Two inputs, both from `filter_tolerance.py --mode designs`: the reference arm's scores and the
device arm's, on the same population rows. Pairing them by row id gives a per-design
device-minus-reference delta, which is the thing pass 7 had to assume was constant because it only
ever had the parity fixture's.

The flip count over the paired rows is the direct answer, and it is not the whole answer: a delta
of d only flips a decision for designs sitting within d of a threshold, so a population of N
designs spread over a range R sees an expected count of about N*d/R flips. With d in the
thousandths that number is below 1 for any affordable N, which means "zero flips observed" has to
be reported together with the margin distribution that produced it, or it says nothing -- the
mistake passes 7 and 8 were careful not to make with their flat ladders. So this also reports, per
criterion, how many designs sit inside one delta of the line, inside ten, and inside a hundred.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

AF2_EASY = {"plddt": (">", 0.8), "i_ptm": (">", 0.5), "i_pae": ("<", 0.35)}


def load(path: Path) -> dict:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def verdict(scalars: dict) -> dict:
    return {k: (scalars[k] > bar if op == ">" else scalars[k] < bar)
            for k, (op, bar) in AF2_EASY.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref, dev = load(Path(args.reference)), load(Path(args.device))
    paired = sorted(set(ref) & set(dev))

    deltas = {k: [] for k in AF2_EASY}
    rows = []
    for rid in paired:
        r, d = ref[rid]["ref"], dev[rid]["ref"]
        rv, dv = verdict(r), verdict(d)
        flipped = sorted(k for k in AF2_EASY if rv[k] != dv[k])
        for k in AF2_EASY:
            deltas[k].append(d[k] - r[k])
        rows.append({"id": rid, "design": ref[rid]["design"], "temp": ref[rid]["temp"],
                     "identity": ref[rid]["identity"],
                     "reference": {k: round(r[k], 6) for k in AF2_EASY},
                     "device": {k: round(d[k], 6) for k in AF2_EASY},
                     "delta": {k: round(d[k] - r[k], 6) for k in AF2_EASY},
                     "accept_reference": all(rv.values()), "accept_device": all(dv.values()),
                     "flipped": flipped})

    # the margin distribution is read off whichever arm has the most rows: it is a property of the
    # population, not of the arm, and the two arms agree on it to within the delta by construction
    pop = dev if len(dev) >= len(ref) else ref
    delta_scale = {k: max((abs(x) for x in deltas[k]), default=0.0) for k in AF2_EASY}
    margins = {}
    for k, (_, bar) in AF2_EASY.items():
        m = sorted(abs(row["ref"][k] - bar) for row in pop.values())
        d = delta_scale[k]
        margins[k] = {
            "n": len(m), "bar": bar, "delta_worst": round(d, 6),
            "min_margin": round(m[0], 6) if m else None,
            "median_margin": round(m[len(m) // 2], 6) if m else None,
            "within_1_delta": sum(1 for x in m if x < d),
            "within_10_delta": sum(1 for x in m if x < 10 * d),
            "within_100_delta": sum(1 for x in m if x < 100 * d),
        }

    flips = sum(1 for r in rows if r["flipped"])
    decision_flips = sum(1 for r in rows if r["accept_reference"] != r["accept_device"])
    lo, hi = wilson(decision_flips, len(rows))
    report = {
        "paired": len(rows),
        "reference_rows": len(ref), "device_rows": len(dev),
        "accept_reference": sum(1 for r in rows if r["accept_reference"]),
        "accept_device": sum(1 for r in rows if r["accept_device"]),
        "criterion_flips": flips,
        "decision_flips": decision_flips,
        "decision_flip_rate": round(decision_flips / len(rows), 4) if rows else None,
        "decision_flip_rate_ci95": [round(lo, 4), round(hi, 4)],
        "delta": {k: {"mean": round(sum(v) / len(v), 6) if v else None,
                      "min": round(min(v), 6) if v else None,
                      "max": round(max(v), 6) if v else None,
                      "worst_abs": round(delta_scale[k], 6)} for k, v in deltas.items()},
        "margins": margins,
        "rows": rows,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
