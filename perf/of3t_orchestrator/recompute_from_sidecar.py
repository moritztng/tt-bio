#!/usr/bin/env python3
"""Recompute a row's published section headlines from another row's per-tensor sidecar.

This exists because it caught two defects in one pass.

D84: I cross-compared two columns of a table whose device column came from other rows on
differently-scoped sets, and published a figure that a three-multiplication subset test refuted.
D85: `of3t-residual` published `atom_attn_dec` at 4.856455e-02 against its own threshold of
4.856425e-02 -- six significant figures of agreement, which is the threshold recomputed rather
than the device measured. Recomputing from `of3t-trajectory`'s sidecar on the IDENTICAL 40 tensors
gave 2.813911e-01, 5.79x higher, reproducing trajectory's own published figure exactly.

Both were catchable only because trajectory published per-tensor `diff_norm` and `ref_norm`
instead of summary statistics -- the campaign's "a result file keeping only extremes cannot be
re-analysed". A mass-weighted headline is then recomputable on any host with no device:

    rel_l2(S) = sqrt( sum_{t in S} diff_norm_t^2 / sum_{t in S} ref_norm_t^2 )

So: whenever two rows report the same quantity, recompute one from the other's per-tensor data.
A disagreement on the same tensor set is a defect in one of them, never a range.

    recompute_from_sidecar.py <sidecar.json> [--expect section=value ...]

Exits non-zero if any --expect disagrees by more than 1e-4 relative, so it can gate a compose.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys


def load(path: str) -> list[dict]:
    rows = json.load(open(path))
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a list of per-tensor rows, got {type(rows).__name__}")
    missing = {"diff_norm", "ref_norm", "section"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path}: rows lack {sorted(missing)} -- a sidecar that keeps only "
                         f"summaries cannot be re-analysed, which is the point of this script")
    return rows


def mass_weighted(rows: list[dict]) -> tuple[float, int, float]:
    num = sum(r["diff_norm"] ** 2 for r in rows)
    den = sum(r["ref_norm"] ** 2 for r in rows)
    if den <= 0:
        raise SystemExit("reference mass is zero over this set -- refusing to divide")
    return math.sqrt(num / den), len(rows), sum(r.get("pct_of_model_mass", 0.0) for r in rows)


SQRT2 = math.sqrt(2.0)


def transpose_signature(rows: list[dict], rel_tol: float = 0.05,
                        cos_tol: float = 0.05) -> list[dict]:
    """Tensors whose gradient was scored against its own TRANSPOSE.

    A transpose is norm-preserving and decorrelating, so dW against dW-transpose reads
    rel_l2 = sqrt(2) with cos = 0 -- independent of dtype, and no precision lever touches it.
    That fingerprint has now caught the same defect class twice in this campaign: D59 (48
    contaminated entries of the per-tensor array) and D86 (a shape test that cannot fire on a
    SQUARE weight, so 87 of 547 tensors were scored against their own transposes and became
    56.22 % of the bounded arm's squared error on 0.2035 % of its mass).

    It is cheap, needs only a per-tensor array, and belongs in every result file's own validation
    rather than in an orchestrator's retrospective -- which is why it lives here beside the
    recomputation.
    """
    out = []
    for r in rows:
        rel, cos = r.get("rel_l2"), r.get("cos")
        if not isinstance(rel, (int, float)) or not isinstance(cos, (int, float)):
            continue
        if abs(rel - SQRT2) < rel_tol and abs(cos) < cos_tol:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar")
    ap.add_argument("--expect", action="append", default=[],
                    help="section=value, or ALL=value for the whole sidecar")
    a = ap.parse_args()
    rows = load(a.sidecar)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["section"]].append(r)

    print(f"{'section':46s} {'n':>5s} {'mass %':>9s} {'mass-weighted rel_l2':>21s}")
    got: dict[str, float] = {}
    for sec in sorted(by, key=lambda s: -len(by[s])):
        v, n, m = mass_weighted(by[sec])
        got[sec] = v
        print(f"{sec:46s} {n:5d} {m:9.4f} {v:21.6e}")
    v, n, m = mass_weighted(rows)
    got["ALL"] = v
    print(f"{'ALL (the whole sidecar)':46s} {n:5d} {m:9.4f} {v:21.6e}")

    rel_tol_used, cos_tol_used = 0.05, 0.05
    sq = transpose_signature(rows, rel_tol_used, cos_tol_used)
    if sq:
        tot = sum(r["diff_norm"] ** 2 for r in rows)
        em = sum(r["diff_norm"] ** 2 for r in sq) / tot * 100 if tot else 0.0
        mm = sum(r["ref_norm"] ** 2 for r in rows)
        ms = sum(r["ref_norm"] ** 2 for r in sq) / mm * 100 if mm else 0.0
        print(f"\nTRANSPOSE SIGNATURE: {len(sq)} of {len(rows)} tensors read rel_l2 ~ sqrt(2) "
              f"with cos ~ 0 -- {em:.2f} % of the squared error on {ms:.4f} % of the mass.")
        print("  That is a gradient scored against its own transpose (D59, D86), not a precision "
              "problem: a transpose is norm-preserving and decorrelating, so no lever moves it.")
        print(f"  Read the {em:.2f} % as a share of THIS arm's error, and do not read a small one "
              f"as harmless: on the")
        print("  shipped diffusion arm these tensors are ~0.01 % of the squared error because the "
              "softmax swamps")
        print("  them, and 56.22 % of it once the softmax bound removes that. A contaminant's "
              "share grows as the")
        print("  dominant error is fixed, so it is exactly the arms you have improved that this "
              "check matters on.")
        print(f"  The count is a LOWER BOUND at this tolerance (|rel-sqrt2|<{rel_tol_used}, "
              f"|cos|<{cos_tol_used}): of3t-residual")
        print("  identified 87 square tensors in this arm by construction, where the fingerprint "
              "catches those whose")
        print("  rel sits close enough to sqrt(2); widen the tolerances to sweep, and check "
              "squareness in the source.")
        for r in sorted(sq, key=lambda r: -r["diff_norm"] ** 2)[:5]:
            print(f"    {str(r.get('param'))[-64:]:66s} rel {r['rel_l2']:.4f} cos {r['cos']:+.4f}")
    else:
        print(f"\nno transpose signature: 0 of {len(rows)} tensors read rel_l2 ~ sqrt(2) with "
              f"cos ~ 0")

    bad = []
    for spec in a.expect:
        if "=" not in spec:
            raise SystemExit(f"--expect wants section=value, got {spec!r}")
        sec, want = spec.rsplit("=", 1)
        if sec not in got:
            bad.append(f"{sec!r} is not a section in this sidecar (have: {sorted(got)})")
            continue
        w = float(want)
        rel = abs(got[sec] - w) / w if w else float("inf")
        verdict = "ok" if rel <= 1e-4 else "DISAGREES"
        print(f"  {verdict:10s} {sec}: recomputed {got[sec]:.6e} vs expected {w:.6e} "
              f"({rel:.2e} relative)")
        if rel > 1e-4:
            bad.append(f"{sec}: recomputed {got[sec]:.6e} but {w:.6e} was published "
                       f"-- {rel:.2e} relative; one of the two is a defect, not a range")
    if bad:
        print("\n".join("FAIL: " + b for b in bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
