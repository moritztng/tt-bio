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

`--rmsd-reference/--rmsd-device` add `af2_easy`'s fourth criterion from `bound_unbound_rmsd.py` and
report the three- and four-criterion numbers side by side. The fourth needs no full-population
coverage to be a bounded statement: `af2_easy` is a conjunction, so it can only turn an accept into
a reject, and a design both arms already reject cannot change the decision-flip count whatever its
RMSD. Only the union of the two arms' three-criterion accept sets can. That is an argument, so it
gets `check_rmsd_coverage` rather than a footnote -- a design accepted on either arm and missing an
RMSD makes the four-criterion number unstated, and it fails loudly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

AF2_EASY = {"plddt": (">", 0.8), "i_ptm": (">", 0.5), "i_pae": ("<", 0.35)}
RMSD_BAR = 3.5      # pxdbench/pxd_configs/eval.py:84-89, thresholded on the value rounded to 2 dp


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


def load_rmsd(path: Path | None) -> dict:
    """id -> bound-unbound RMSD, unrounded, from bound_unbound_rmsd.py."""
    if not path:
        return {}
    return {r["id"]: r["bound_unbound_rmsd"]
            for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


def passes_rmsd(rmsd: float | None) -> bool:
    """Upstream rounds to two decimals before comparing (main_af2_monomer.py:164)."""
    return rmsd is not None and round(rmsd, 2) < RMSD_BAR


def accept3(scalars: dict) -> bool:
    return all(verdict(scalars).values())


def accept4(scalars: dict, rmsd: float | None) -> bool:
    return accept3(scalars) and passes_rmsd(rmsd)


def check_rmsd_coverage(ref: dict, dev: dict, rmsd_ref: dict, rmsd_dev: dict) -> dict:
    """The conjunction bounds which designs need an RMSD; this enforces that bound.

    A design both arms reject under three criteria stays reject-on-both under four whatever its
    RMSD, so it needs none. Every design accepted on either arm needs one on both arms, or the
    four-criterion accept sets are not computable and the report would silently fall back to the
    three-criterion ones.
    """
    union, gaps = [], []
    for rid in sorted(set(ref) & set(dev)):
        if not (accept3(ref[rid]["ref"]) or accept3(dev[rid]["ref"])):
            continue
        union.append(rid)
        for arm, table in (("reference", rmsd_ref), ("device", rmsd_dev)):
            if rid not in table:
                gaps.append("%s/%s" % (rid, arm))
    assert not gaps, ("no bound-unbound RMSD for designs accepted on at least one arm, so the "
                      "four-criterion verdict is unstated for them: %s" % sorted(gaps))
    return {"union_of_three_criterion_accepts": union, "n_union": len(union),
            "reject_on_both_no_rmsd_needed": len(set(ref) & set(dev)) - len(union)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--envelope", default=None,
                    help="a third arm inside the reference's own precision freedom (float32 trunk "
                         "instead of bfloat16). Its delta is the bar the device delta is judged "
                         "against: a device miss smaller than it is not the port's to fix.")
    ap.add_argument("--rmsd-reference", default=None,
                    help="bound_unbound_rmsd.py output for the reference arm: adds af2_easy's "
                         "fourth criterion to the report")
    ap.add_argument("--rmsd-device", default=None, help="the same for the device arm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref, dev = load(Path(args.reference)), load(Path(args.device))
    rmsd_ref = load_rmsd(Path(args.rmsd_reference) if args.rmsd_reference else None)
    rmsd_dev = load_rmsd(Path(args.rmsd_device) if args.rmsd_device else None)
    paired = sorted(set(ref) & set(dev))

    deltas = {k: [] for k in AF2_EASY}
    rows = []
    for rid in paired:
        r, d = ref[rid]["ref"], dev[rid]["ref"]
        rv, dv = verdict(r), verdict(d)
        flipped = sorted(k for k in AF2_EASY if rv[k] != dv[k])
        for k in AF2_EASY:
            deltas[k].append(d[k] - r[k])
        row = {"id": rid, "design": ref[rid]["design"], "temp": ref[rid]["temp"],
               "identity": ref[rid]["identity"],
               "reference": {k: round(r[k], 6) for k in AF2_EASY},
               "device": {k: round(d[k], 6) for k in AF2_EASY},
               "delta": {k: round(d[k] - r[k], 6) for k in AF2_EASY},
               "accept_reference": all(rv.values()), "accept_device": all(dv.values()),
               "flipped": flipped}
        if rmsd_ref or rmsd_dev:
            rr, rd = rmsd_ref.get(rid), rmsd_dev.get(rid)
            row["rmsd_reference"] = rr
            row["rmsd_device"] = rd
            row["rmsd_delta"] = round(rd - rr, 6) if (rr is not None and rd is not None) else None
            row["accept_reference_4"] = accept4(r, rr)
            row["accept_device_4"] = accept4(d, rd)
            row["flipped_4"] = sorted(flipped + (["bound_unbound_rmsd"]
                                                 if passes_rmsd(rr) != passes_rmsd(rd) else []))
        rows.append(row)

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

    envelope = None
    if args.envelope:
        env = load(Path(args.envelope))
        shared = sorted(set(ref) & set(env))
        erows = []
        for rid in shared:
            r, e = ref[rid]["ref"], env[rid]["ref"]
            rv, ev = verdict(r), verdict(e)
            d = dev[rid]["ref"] if rid in dev else None
            erows.append({
                "id": rid,
                "reference_bf16": {k: round(r[k], 6) for k in AF2_EASY},
                "reference_fp32": {k: round(e[k], 6) for k in AF2_EASY},
                "envelope_delta": {k: round(e[k] - r[k], 6) for k in AF2_EASY},
                "device_delta": {k: round(d[k] - r[k], 6) for k in AF2_EASY} if d else None,
                # >1 means the device sits outside the reference's own precision freedom
                "device_over_envelope": {
                    k: (round(abs(d[k] - r[k]) / abs(e[k] - r[k]), 3)
                        if d and abs(e[k] - r[k]) > 0 else None) for k in AF2_EASY},
                "accept_reference_bf16": all(rv.values()),
                "accept_reference_fp32": all(ev.values()),
                "flipped_by_envelope": sorted(k for k in AF2_EASY if rv[k] != ev[k]),
            })
        efl = sum(1 for r in erows if r["accept_reference_bf16"] != r["accept_reference_fp32"])
        elo, ehi = wilson(efl, len(erows))
        envelope = {
            "n": len(erows),
            "criterion_flips": sum(1 for r in erows if r["flipped_by_envelope"]),
            "decision_flips": efl,
            "decision_flip_rate": round(efl / len(erows), 4) if erows else None,
            "decision_flip_rate_ci95": [round(elo, 4), round(ehi, 4)],
            "worst_abs": {k: round(max(abs(r["envelope_delta"][k]) for r in erows), 6)
                          for k in AF2_EASY} if erows else None,
            "rows": erows,
        }

    four = None
    if rmsd_ref or rmsd_dev:
        coverage = check_rmsd_coverage(ref, dev, rmsd_ref, rmsd_dev)
        f4 = [r for r in rows if r["accept_reference_4"] != r["accept_device_4"]]
        f3 = [r for r in rows if r["accept_reference"] != r["accept_device"]]
        lo4, hi4 = wilson(len(f4), len(rows))
        four = {
            "coverage": coverage,
            "rmsd_rows": {"reference": len(rmsd_ref), "device": len(rmsd_dev)},
            "bar": RMSD_BAR,
            "accept_reference": sum(1 for r in rows if r["accept_reference_4"]),
            "accept_device": sum(1 for r in rows if r["accept_device_4"]),
            "criterion_flips": sum(1 for r in rows if r["flipped_4"]),
            "decision_flips": len(f4),
            "decision_flip_rate": round(len(f4) / len(rows), 4) if rows else None,
            "decision_flip_rate_ci95": [round(lo4, 4), round(hi4, 4)],
            "flip_ids": [r["id"] for r in f4],
            # a criterion that only removes accepts can only remove flips, so these two name the
            # whole difference between the two headline numbers
            "flips_removed_vs_three": [r["id"] for r in f3 if r not in f4],
            "flips_added_vs_three": [r["id"] for r in f4 if r not in f3],
            "rmsd_accepts_removed": {
                "reference": [r["id"] for r in rows
                              if r["accept_reference"] and not r["accept_reference_4"]],
                "device": [r["id"] for r in rows
                           if r["accept_device"] and not r["accept_device_4"]]},
        }
        for arm, table in (("reference", rmsd_ref), ("device", rmsd_dev)):
            vals = sorted(table.values())
            if not vals:
                continue
            deltas4 = [abs(r["rmsd_delta"]) for r in rows if r.get("rmsd_delta") is not None]
            d = max(deltas4, default=0.0)
            margins["bound_unbound_rmsd/" + arm] = {
                "n": len(vals), "bar": RMSD_BAR, "delta_worst": round(d, 6),
                "min": round(vals[0], 6), "median": round(vals[len(vals) // 2], 6),
                "max": round(vals[-1], 6),
                "min_margin": round(min(abs(v - RMSD_BAR) for v in vals), 6),
                "median_margin": round(sorted(abs(v - RMSD_BAR) for v in vals)[len(vals) // 2], 6),
                "within_1_delta": sum(1 for v in vals if abs(v - RMSD_BAR) < d),
                "within_10_delta": sum(1 for v in vals if abs(v - RMSD_BAR) < 10 * d),
                "within_100_delta": sum(1 for v in vals if abs(v - RMSD_BAR) < 100 * d),
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
        "four_criterion": four,
        "reference_precision_envelope": envelope,
        "rows": rows,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
