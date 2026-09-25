"""Device trajectory against the JAX reference trajectory, on the columns BindCraft 2 records.

BCX has run device trajectories for days and never once put one beside the reference binder the
same unedited `examples/pdl1.json` produced. Stage lines in a log are not that comparison: a stage
line carries two numbers, `!_Trajectories.csv` carries thirty-nine, and the acceptance filters are
defined on the second set.

Two things this prints that reading either CSV alone does not give you:

1. THE TRAJECTORY ROW IS NOT SCORED AGAINST THE FULL FILTER SET. Of the seven filters BindCraft 2
   prints at startup, `Binder_RMSD` and `Unbound_Binder_pLDDT` exist only on an MPNN candidate
   (`2_Refolded/!_Refolded.csv`) because both need the binder refolded alone. Applying all seven
   to a trajectory row silently passes a design on two filters that were never evaluated. This
   script applies only the five a trajectory row can answer and names the other two.

2. THE PRINTED THRESHOLD IS NOT THE EFFECTIVE BAR. `MPNN_stage.predict_validation_ensemble` folds
   the validation models one at a time and tests the confidence filters against EACH MODEL
   SEPARATELY, while the value written to the CSV is `ensemble_mean_predictions`, the arithmetic
   mean over the models folded. So a candidate recorded at i_pTM 0.75 is rejected `failed [i_pTM]`
   against a printed `i_pTM >= 0.7`, which is exactly what the reference run did twice. An
   acceptance count derived by applying the printed thresholds to a recorded row is an over-count,
   and this script says so rather than pretending the row settles it.

Usage: stage_compare.py DEVICE_PROJECT_DIR REFERENCE_SNAPSHOT_DIR
  DEVICE_PROJECT_DIR      an arm project folder holding 1_Trajectories/!_Trajectories.csv
  REFERENCE_SNAPSHOT_DIR  a snapshot of the JAX run holding _Trajectories.csv and _Refolded.csv
"""
import csv
import os
import sys

# Printed by BindCraft 2 at startup for the unedited examples/pdl1.json, both runs, verbatim.
FILTERS = [("Backbone_Clashes", "<=", 0.0), ("Binder_RMSD", "<=", 3.5),
           ("Interface_Residues", ">=", 7.0), ("Unbound_Binder_pLDDT", ">=", 0.7),
           ("i_pAE", "<=", 0.35), ("i_pTM", ">=", 0.7), ("pTM", ">=", 0.55)]

# Higher is better for these, lower for the rest. Used only to label the delta column.
HIGHER = {"i_pTM", "pLDDT", "pTM", "Target_pLDDT", "Unbound_Binder_pLDDT", "Interface_Residues",
          "Hotspot_Contact_Fraction", "Interface_BuriedArea", "Interface_BuriedArea_Fraction",
          "SS_pLDDT", "Epitope_Residues_Contacted", "Binder_Helix_Fraction"}

# Reported per trajectory but set by the draw, not by how well the design went.
DRAW = {"length", "Binder_Mass_kDa", "Binder_Extinction", "Binder_pI", "Binder_Net_Charge"}

SKIP = {"design", "trajectory", "rank", "Binder_Sequence", "bindcraft_version", "Timing", "hash",
        "terminated", "autotuned", "Interface_Binder_Residues", "Interface_Target_Residues",
        "outcome", "failed_filters"}


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def table(root, names):
    for name in names:
        found = rows(os.path.join(root, name))
        if found:
            return found
    return []


def ledger(root):
    return table(root, ("1_Trajectories/!_Trajectories.csv", "_Trajectories.csv"))


def refolded(root):
    return table(root, ("2_Refolded/!_Refolded.csv", "_Refolded.csv"))


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def holds(value, comparison, threshold):
    return value <= threshold if comparison == "<=" else value >= threshold


def filter_report(label, row, candidates):
    """Apply the printed filter set, separating what this row can answer from what it cannot."""
    candidate_columns = set(candidates[0]) if candidates else set()
    print("  %s" % label)
    for name, comparison, threshold in FILTERS:
        value = number(row.get(name))
        if value is None:
            where = "candidate-level only" if name in candidate_columns else "not recorded"
            print("    %-22s %2s %-6g  NOT EVALUABLE on a trajectory row (%s)"
                  % (name, comparison, threshold, where))
            continue
        print("    %-22s %2s %-6g  %-8g %s"
              % (name, comparison, threshold, value,
                 "pass" if holds(value, comparison, threshold) else "FAIL"))


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    device_root, reference_root = argv
    device, reference = ledger(device_root), ledger(reference_root)
    if not device or not reference:
        print("need a trajectory ledger on both sides; device %d rows, reference %d rows"
              % (len(device), len(reference)))
        return 1
    dev, ref = device[-1], reference[0]
    dev_candidates, ref_candidates = refolded(device_root), refolded(reference_root)

    print("device    %s  (%s)" % (dev["design"], device_root))
    print("reference %s  (%s)" % (ref["design"], reference_root))
    print()
    print("NOT A MATCHED PAIR: different binder draws, different lengths (%s aa against %s aa),"
          % (dev.get("length"), ref.get("length")))
    print("different seeds. Every number below is one device sample beside one reference sample.")
    print()
    print("  %-30s %12s %12s %10s" % ("metric", "device", "reference", "delta"))
    print("  " + "-" * 68)
    for name in dev:
        if name in SKIP or name not in ref:
            continue
        left, right = number(dev[name]), number(ref[name])
        if left is None or right is None:
            continue
        note = "  (draw)"
        if name not in DRAW:
            note = ""
            if abs(left - right) > 1e-9:
                better = (left > right) if name in HIGHER else (left < right)
                note = "  device better" if better else "  reference better"
        print("  %-30s %12g %12g %+10g%s" % (name, left, right, left - right, note))

    print()
    print("acceptance filters, as BindCraft 2 prints them for this example")
    filter_report("device", dev, dev_candidates)
    filter_report("reference trajectory", ref, ref_candidates)
    print()
    print("  Two of the seven are unanswerable from a trajectory row. They are answered on the")
    print("  MPNN candidates, and the device trajectory has %d of those against the reference's %d."
          % (len(dev_candidates), len(ref_candidates)))
    if ref_candidates:
        passed = [c for c in ref_candidates if c.get("outcome") == "passed"]
        print("  reference candidates: %d scored, %d passed, failures %s"
              % (len(ref_candidates), len(passed),
                 sorted({c["failed_filters"] for c in ref_candidates if c.get("failed_filters")})))
        for candidate in ref_candidates:
            value = number(candidate.get("i_pTM"))
            if candidate.get("failed_filters") == "i_pTM" and value is not None and value >= 0.7:
                print("  %s recorded i_pTM %.2f, above the printed 0.7, and still failed i_pTM."
                      % (candidate["design"].rpartition("_")[2], value))
                print("  The filter ran per validation model; the CSV holds those models' mean.")
                break
    if not dev_candidates:
        print("  The device trajectory produced NO candidate rows, so it was never scored against")
        print("  the filter set at all. It is not an acceptance and it is not a rejection.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
