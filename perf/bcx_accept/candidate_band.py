"""Every scored MPNN candidate on both arms, so the comparison is not 1 against 1.

`ranked_compare.py` compares the two designs BindCraft 2 KEPT, which is one row per trajectory and
therefore one sample each side. Every candidate that reached `2_Refolded/!_Refolded.csv` was folded
through the same validation ensemble and tested against the same filters, accepted or not, so the
candidate rows are the larger sample available without spending another trajectory.

They are NOT independent samples. All candidates within a trajectory are ProteinMPNN redesigns of
one hallucinated backbone, so they share it: this compares two clusters of correlated rows, and it
answers "do the device candidates sit inside the reference's band" rather than "are the two
distributions the same". A candidate outside the band on a filter is worth reading; one inside is
not evidence of agreement.

Usage: candidate_band.py DEVICE_REFOLDED_CSV REFERENCE_REFOLDED_CSV
"""
import csv
import sys

# The filters a refolded candidate is actually rejected on, with the direction that passes.
# Floors from the acceptance set BindCraft 2 prints at the top of a design run.
FILTERS = [("i_pTM", ">=", 0.70), ("pTM", ">=", 0.55), ("i_pAE", "<=", 0.35),
           ("Unbound_Binder_pLDDT", ">=", 0.70), ("Binder_RMSD", "<=", 3.5),
           ("Interface_Residues", ">=", 7), ("Backbone_Clashes", "<=", 0)]


def rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def values(rs, field):
    out = []
    for r in rs:
        try:
            out.append(float(r[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return sorted(out)


def band(vals):
    return "%.2f-%.2f" % (vals[0], vals[-1]) if vals else "-"


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    dev, ref = rows(argv[0]), rows(argv[1])
    passed = lambda rs: sum(1 for r in rs if r.get("outcome") == "passed")
    print("device    %d candidates scored, %d passed" % (len(dev), passed(dev)))
    print("reference %d candidates scored, %d passed" % (len(ref), passed(ref)))
    print("  correlated within each trajectory: MPNN redesigns of one backbone, not %d and %d"
          % (len(dev), len(ref)))
    print()
    print("%-24s %5s %6s %14s %14s  %s"
          % ("filter", "pass", "floor", "device", "reference", "device vs band"))
    for field, direction, floor in FILTERS:
        d, r = values(dev, field), values(ref, field)
        if not d or not r:
            continue
        if d[0] >= r[0] and d[-1] <= r[-1]:
            where = "inside"
        elif (d[-1] < r[0]) or (d[0] > r[-1]):
            where = "DISJOINT"
        else:
            better = (d[-1] > r[-1]) if direction == ">=" else (d[0] < r[0])
            where = "wider, %s end" % ("better" if better else "worse")
        print("%-24s %5s %6s %14s %14s  %s"
              % (field, direction, floor, band(d), band(r), where))
    print()
    print("A band that is inside is not agreement, only an absence of an outlier.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
