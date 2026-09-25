#!/usr/bin/env python3
"""Does the conditioning pin one chain relative to the other, or is it free?

This is the half of the pxdesign 1536 question that needs no device, and for one of the two
targets it is a proof rather than a measurement.

pxdesign sees the target only as a 64-bin distogram over 2-22 A
(`tt_bio/pxdesign/featurize.py:40`). Bins are (22-2)/63 = 0.317 A wide, so a pair the model can
resolve constrains that distance to about +/-0.16 A, and a pair beyond 22 A constrains nothing
at all. For a two-chain target the question is whether the INTER-chain pairs pin the six rigid
degrees of freedom of one chain against the other.

Measured by moving chain B rigidly and counting how many inter-chain pairs leave the bin they
were in. Two outcomes matter and they are qualitatively different:

  * pairs change -> the placement is constrained, and how small a motion it takes says how
    tightly.
  * NO pair changes -> the input is INVARIANT under that motion. The relative placement is then
    not a function of the input, so no model and no algorithm can recover it, and a fit that
    tries lands at the scale of the separation. That is not a defect in the model.

    python3 perf/mgxaccuracy/rigidity.py            # both ladder targets at their 1536 crop
"""
import argparse
import importlib.util
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perf.mgxaccuracy.contact import atoms, TEMPL_TOP_A  # noqa: E402

BIN_MIN, BIN_MAX, NBINS = 2.0, 22.0, 64


def _ladder():
    spec = importlib.util.spec_from_file_location("ld", ROOT / "perf/bhdesign/ladder.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rot(axis, deg):
    import numpy as np
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def run(label: str, cif: pathlib.Path) -> dict:
    import numpy as np
    bound = np.linspace(BIN_MIN, BIN_MAX, NBINS - 1)
    width = (BIN_MAX - BIN_MIN) / (NBINS - 1)

    def binof(d):
        return (d[..., None] > bound).sum(-1)

    meta, xyz = atoms(cif)
    ca = [(c, p) for (c, _s, a), p in zip(meta, xyz) if a == "CA"]
    ch = np.array([c for c, _ in ca])
    P = np.array([p for _, p in ca])
    chains = sorted(set(ch.tolist()))
    if len(chains) < 2:
        print(f"{label}: single chain, no relative placement to pin")
        return {"label": label, "chains": chains}
    A, B = P[ch == chains[0]], P[ch == chains[1]]
    d0 = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    resolvable = d0 <= TEMPL_TOP_A
    nres = int(resolvable.sum())
    b0 = binof(d0)
    print(f"\n{label}: chains {chains[0]}={len(A)} {chains[1]}={len(B)}, inter-chain pairs "
          f"{d0.size}, resolvable (<= {TEMPL_TOP_A} A) {nres} ({nres / d0.size * 100:.2f}%), "
          f"bin width {width:.3f} A")
    print(f"   {'motion of chain ' + chains[1]:<24}{'pairs changing bin':>20}{'of resolvable':>15}")
    cen = B.mean(0)
    out = {}
    motions = [("translate 0.1 A", B + np.array([0.1, 0, 0])),
               ("translate 0.5 A", B + np.array([0.5, 0, 0])),
               ("translate 2.0 A", B + np.array([2.0, 0, 0])),
               ("translate 10 A", B + np.array([10.0, 0, 0])),
               ("rotate 1 deg", (B - cen) @ rot([0, 0, 1], 1).T + cen),
               ("rotate 5 deg", (B - cen) @ rot([0, 0, 1], 5).T + cen),
               ("rotate 30 deg", (B - cen) @ rot([0, 0, 1], 30).T + cen)]
    for name, Bp in motions:
        d1 = np.linalg.norm(A[:, None, :] - Bp[None, :, :], axis=-1)
        changed = int(((binof(d1) != b0) & resolvable).sum())
        out[name] = changed
        frac = f"{changed / nres * 100:.1f}%" if nres else "n/a"
        print(f"   {name:<24}{changed:>20}{frac:>15}")
    if nres == 0:
        print(f"   -> the conditioning is INVARIANT under every motion above. The relative "
              f"placement of {chains[1]} is not a function of this input.")
    return {"label": label, "resolvable": nres, "changed": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--targets", nargs="+", default=[
        str(ROOT / "perf/mgxaccuracy/targets/gpb_dimer_1646.cif"),
        str(ROOT / "perf/bhdesign/targets/big_1831.cif")])
    a = ap.parse_args()
    ld = _ladder()
    w = pathlib.Path(tempfile.mkdtemp())
    for t in a.targets:
        ld.boltzgen_fixture(w, a.size, pathlib.Path(t))
        run(f"{pathlib.Path(t).name} {a.size} crop", w / f"bgt{a.size}.cif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
