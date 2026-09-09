"""Negative control for struct_signal: mutate one thing and require the number to move.

Four arms. The first two are the controls the instrument was calibrated with, re-run here
rather than remembered. The last two are the ones it did NOT have, which is why both of
`check_structure.py`'s exclusions could go missing from it unnoticed: a control measured on a
deposited crystal structure cannot catch a rule about model output, because a deposit carries no
V0..V8 placeholders and its disulfides sit at 2.05 A, just outside the 2.0 A clash cutoff.

    python perf/ceilings/struct_signal_control.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from struct_signal import clashes, continuity, read_cif  # noqa: E402

ART = pathlib.Path(__file__).resolve().parents[1] / "wh-correctness/results/artifacts"
FOLD = ART / "comp_abag_boltz2/target_1.cif"          # 2 chains, 4913 atoms, has disulfides
RFD3 = ART / "des_rfd3_binder/design_0.cif"           # 964 of 2051 atoms are V0..V8


def arm(name, ok, got):
    print(f"  {'caught' if ok else 'NOT CAUGHT'}: {name} -- {got}")
    return ok


def main():
    cols = read_cif(FOLD)
    atom, comp, asym, seq, elem, xyz = cols
    passed = []

    # 1. Break the backbone. Continuity must see it; clashes must not care.
    # Cut inside a chain, not at an absolute residue number: label_seq_id restarts per chain,
    # so a fixed threshold above the longest chain selects nothing and the arm passes vacuously.
    #
    # Translate the tail clear OUT of the structure rather than 20 A across it. A short shift
    # breaks the backbone and also rams the tail into the rest of the fold, so it moves both
    # numbers and cannot show the two arms are independent -- measured, clash_frac 0.00081 ->
    # 0.05455 at 20 A. Far enough away and only continuity can see it.
    shifted = xyz.copy()
    shifted[seq > int(np.median(seq[seq > 0]))] += 500.0
    base_w, base_b, _ = continuity(*cols)
    w, b, _ = continuity(atom, comp, asym, seq, elem, shifted)
    passed.append(arm("a backbone shift out of the box moves ca_breaks",
                      b > base_b and w > base_w,
                      f"worst {base_w:.2f} -> {w:.2f}, breaks {base_b} -> {b}"))
    f0, _ = clashes(*cols)
    f1, _ = clashes(atom, comp, asym, seq, elem, shifted)
    passed.append(arm("and leaves clash_frac alone (the two arms are independent)",
                      abs(f1 - f0) < 5e-4, f"clash_frac {f0:.5f} -> {f1:.5f}"))

    # 2. Superpose one chain on another. Clashes must see it; continuity must not care.
    ch = np.unique(asym)
    stacked = xyz.copy()
    m = asym == ch[0]
    stacked[m] += xyz[asym == ch[-1]].mean(0) - xyz[m].mean(0)
    f2, _ = clashes(atom, comp, asym, seq, elem, stacked)
    _, b2, _ = continuity(atom, comp, asym, seq, elem, stacked)
    passed.append(arm("superposing two chains moves clash_frac, not continuity",
                      f2 > f0 * 10 and b2 == base_b,
                      f"clash_frac {f0:.5f} -> {f2:.5f}, breaks {base_b} -> {b2}"))

    # 3. The disulfide exclusion. Rename the cysteines and the same file must score WORSE:
    # SG-SG pairs the rule was hiding come back as clashes. If renaming changes nothing the
    # exclusion is not running.
    comp_x = np.where((comp == "CYS") & (atom == "SG"), "XXX", comp)
    f3, _ = clashes(atom, comp_x, asym, seq, elem, xyz)
    passed.append(arm("disulfides are excluded (un-naming the cysteines re-reports them)",
                      f3 > f0, f"clash_frac {f0:.5f} CYS -> {f3:.5f} un-named"))

    # 4. The virtual-atom exclusion, on RFD3 output. Rename V0..V8 to a real carbon name and
    # both the count and the denominator must jump.
    ratom, rcomp, rasym, rseq, relem, rxyz = read_cif(RFD3)
    virt = np.array([a.startswith("V") and a[1:].isdigit() for a in ratom], bool)
    f4, n4 = clashes(ratom, rcomp, rasym, rseq, relem, rxyz)
    f5, n5 = clashes(np.where(virt, "CB", ratom), rcomp, rasym, rseq, relem, rxyz)
    passed.append(arm("RFD3 V0..V8 placeholders are excluded from numerator and denominator",
                      n5 > n4 and f5 > f4,
                      f"heavy {n4} -> {n5}, clash_frac {f4:.5f} -> {f5:.5f} ({int(virt.sum())} placeholders)"))

    print(f"\n{sum(passed)}/{len(passed)} invariants demonstrated to fail when violated")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
