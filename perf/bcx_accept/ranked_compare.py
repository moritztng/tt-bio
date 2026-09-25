"""Two ACCEPTED designs side by side, off `3_Ranked/!_Ranked.csv` on each arm.

`stage_compare.py` compares trajectory rows, which is what was available before any device
trajectory reached the acceptance filters. A trajectory row cannot carry `Binder_RMSD` or
`Unbound_Binder_pLDDT` -- both need the binder refolded alone -- so two of the seven acceptance
filters are simply absent from it. This script compares the designs BindCraft 2 KEPT, where all
seven exist and both arms were judged by the same program: on the shipped `examples/pdl1.json` the
validation pair is `model_1_ptm`/`model_2_ptm`, which is not card-resident, so the device arm's
candidates fold on BindCraft 2's own JAX exactly as the reference's do. The design loop differs;
the filter does not.

It is not a matched pair unless the two draws agree, so the header prints length, seed and draw
hash and leaves the reading to whoever runs it.

Usage: ranked_compare.py DEVICE_RANKED_CSV REFERENCE_RANKED_CSV
"""
import csv
import sys

# Filters BindCraft 2 rejects a candidate on, from the acceptance set the run log prints. Shown
# first because they decide acceptance; everything else is descriptive.
FILTERS = ["Backbone_Clashes", "Binder_RMSD", "Interface_Residues", "Unbound_Binder_pLDDT",
           "i_pAE", "i_pTM", "pTM"]

# Higher is better for these; the rest are lower-is-better or descriptive.
HIGHER = {"i_pTM", "pLDDT", "pTM", "Unbound_Binder_pLDDT", "Target_pLDDT", "Interface_Residues",
          "Hotspot_Contact_Fraction", "Interface_BuriedArea", "Interface_BuriedArea_Fraction",
          "SS_pLDDT", "Epitope_Residues_Contacted"}
LOWER = {"i_pDAE", "i_pAE", "Binder_RMSD", "Target_RMSD", "Off_Epitope_Contact_Fraction",
         "Backbone_Clashes", "All_Atom_Clashes"}
SKIP = {"rank", "design", "hash", "Binder_Sequence", "Interface_Binder_Residues",
        "Interface_Target_Residues"}


def top(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit("no ranked design in %s" % path)
    return rows[0]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def verdict(field, a, b):
    if a is None or b is None or a == b:
        return "tie" if a == b else ""
    if field in HIGHER:
        return "device" if a > b else "reference"
    if field in LOWER:
        return "device" if a < b else "reference"
    return ""


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    dev, ref = top(argv[0]), top(argv[1])
    print("device    %s  (%s aa)" % (dev["design"], dev["length"]))
    print("reference %s  (%s aa)" % (ref["design"], ref["length"]))
    if dev["length"] != ref["length"]:
        print("DIFFERENT DRAWS -- lengths %s and %s. Interface size scales with binder length, so"
              % (dev["length"], ref["length"]))
        print("read Interface_Residues and Interface_BuriedArea against the FRACTION beside them.")
    fields = [f for f in FILTERS if f in dev and f in ref]
    fields += [f for f in dev if f in ref and f not in SKIP and f not in fields]
    print()
    print("%-32s %10s %10s  %s" % ("field", "device", "reference", "better"))
    wins = {"device": 0, "reference": 0}
    for f in fields:
        a, b = number(dev[f]), number(ref[f])
        who = verdict(f, a, b)
        if who in wins:
            wins[who] += 1
        print("%-32s %10s %10s  %s%s"
              % (f, dev[f], ref[f], who, "   <- acceptance filter" if f in FILTERS else ""))
    print()
    print("device better on %d, reference better on %d, of %d comparable fields."
          % (wins["device"], wins["reference"], len(fields)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
