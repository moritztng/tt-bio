"""#17's deliverable table, read out of BindCraft 2's own summary.csv.

    python3 perf/bci_accept/bci17_report.py card=<proj_dir> host=<proj_dir>

`summary.csv` carries one row per (scope, metric) with samples/mean/std/min/max. Scope
`candidate` is the ProteinMPNN redesign rows, which is where #17's signature lives: before
`c79d1d0e5` the on-card arm put `Target_pLDDT` at 0.34-0.41 against host JAX's 0.92-0.93 and
`Interface_Residues` at the binder length on every row. That column is 10 rows per trajectory and
moves by an order of magnitude, which is why it, and not the acceptance count, is the evidence.

Ranges are quoted as min-max because that is the form the comparison was made in. Accepted counts
come from `3_Ranked/!_Ranked.csv`, BindCraft 2's own ranked table, which is what the reporter
quotes his 0 of 8 and 3 of 8 from -- not from anything recomputed here.
"""
import argparse
import csv
import pathlib
import sys

SIGNATURE = ("Target_pLDDT", "Interface_Residues", "Hotspot_Contact_Fraction", "i_pTM")


def read(proj):
    """-> ({(scope, metric): row}, accepted_count_or_None)."""
    proj = pathlib.Path(proj)
    rows = {}
    summary = proj / "summary.csv"
    if summary.exists():
        with open(summary) as fh:
            for r in csv.DictReader(fh):
                rows[(r["scope"], r["metric"])] = r
    ranked = proj / "3_Ranked" / "!_Ranked.csv"
    accepted = None
    if ranked.exists():
        with open(ranked) as fh:
            accepted = sum(1 for _ in csv.DictReader(fh))
    return rows, accepted


def rng(row):
    if row is None:
        return "-"
    lo, hi = float(row["min"]), float(row["max"])
    return f"{lo:.2f}" if lo == hi else f"{lo:.2f}-{hi:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", metavar="name=project_dir")
    ap.add_argument("--trajectories", type=int, default=8)
    a = ap.parse_args()
    arms = []
    for spec in a.arms:
        if "=" not in spec:
            sys.exit(f"expected name=project_dir, got {spec!r}")
        name, _, proj = spec.partition("=")
        arms.append((name, *read(proj)))

    w = max(len(n) for n, _, _ in arms) + 2
    print("accepted, from BindCraft 2's own ranked table")
    for name, _, acc in arms:
        print(f"  {name:<{w}} {'?' if acc is None else acc} of {a.trajectories}")

    print("\nredesign candidates (scope=candidate), min-max")
    print(f"  {'metric':<26}" + "".join(f"{n:<16}" for n, _, _ in arms))
    for m in SIGNATURE:
        line = f"  {m:<26}"
        for _, rows, _ in arms:
            line += f"{rng(rows.get(('candidate', m))):<16}"
        print(line)
    n = [rows.get(("candidate", "i_pTM")) for _, rows, _ in arms]
    print(f"  {'(candidate rows)':<26}" + "".join(
        f"{(r['samples'] if r else '-'):<16}" for r in n))

    # Deliberately NOT a per-stage table. summary.csv's stage scopes carry the loss recorder's
    # per-round values (`<target>.iptm`), and `judge_stage` folds separately: paired_table.py
    # documents a trajectory whose losses.csv ended `screen` on i_pTM 0.19 while the printed line
    # said 0.24. The printed line is what the reporter's own table is made of, so per-stage
    # numbers come from paired_table.py reading the LOGS, not from here.
    print()
    print("per-stage i_pTM: use paired_table.py on the arm logs, not summary.csv")
    print("  (judge_stage folds separately from the loss recorder; the two disagree)")


if __name__ == "__main__":
    main()
