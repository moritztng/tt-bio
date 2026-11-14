"""Score a tt-bio RFD4 feature dict against an upstream golden, field by field.

This row's bar is EXACT, not close: the host path produces the same features, the same masks
and the same dtypes as upstream, or it is wrong. Floats are compared bit for bit by default
because featurization is gather/scatter and table lookup, not arithmetic that could legitimately
reassociate; `--atol` exists to localise a discrepancy while debugging, never to pass a run.

    python3 score_features.py <golden.npz> <ours.npz> [--atol 0] [--quiet]

Exit status is 0 only when every field in the golden is present, same shape, same dtype and
equal. A field we produce that the golden does not have is reported too: an extra feature is a
divergence from upstream just as a missing one is.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def compare(golden: dict, ours: dict, atol: float = 0.0):
    """Return (findings, n_equal). A finding is (field, kind, detail)."""
    findings = []
    equal = 0
    for key in sorted(golden):
        g = golden[key]
        if key not in ours:
            findings.append((key, "missing", f"golden has {g.shape} {g.dtype}"))
            continue
        o = ours[key]
        if g.shape != o.shape:
            findings.append((key, "shape", f"golden {g.shape} vs ours {o.shape}"))
            continue
        if g.dtype != o.dtype:
            findings.append((key, "dtype", f"golden {g.dtype} vs ours {o.dtype}"))
            continue
        if g.dtype.kind in "fc":
            gnan, onan = np.isnan(g), np.isnan(o)
            if not np.array_equal(gnan, onan):
                findings.append((key, "nan", f"{int((gnan ^ onan).sum())} positions differ in NaN-ness"))
                continue
            d = np.abs(g[~gnan] - o[~onan]) if gnan.any() else np.abs(g - o)
            worst = float(d.max()) if d.size else 0.0
            if worst > atol:
                n = int((d > atol).sum())
                findings.append((key, "value", f"{n}/{d.size} differ, max |diff| {worst:.3e}"))
                continue
        elif not np.array_equal(g, o):
            n = int((g != o).sum())
            findings.append((key, "value", f"{n}/{g.size} differ"))
            continue
        equal += 1
    for key in sorted(set(ours) - set(golden)):
        findings.append((key, "extra", f"ours has {ours[key].shape} {ours[key].dtype}"))
    return findings, equal


def load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden", type=Path)
    ap.add_argument("ours", type=Path)
    ap.add_argument("--atol", type=float, default=0.0,
                    help="debugging aid only; a passing run is scored at 0")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    findings, equal = compare(load(a.golden), load(a.ours), a.atol)
    total = equal + len([f for f in findings if f[1] != "extra"])
    if not a.quiet or findings:
        print(f"{equal}/{total} fields exact" + (f" (atol {a.atol:g})" if a.atol else ""))
        for field, kind, detail in findings:
            print(f"  {kind.upper():8s} {field}: {detail}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
