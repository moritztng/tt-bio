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
