"""CA-lDDT and CA RMSD of cdk2x2_512 folds against the experimental 1HCL, per pseudo-domain, arm against arm.

    python perf/spd_overhead/hcl_native.py BASE_RUN TEST_RUN [--out OUT.json]

Each RUN is a perf/spd/bench.py --out directory holding struct_cdk2x2_512_s<seed>/. The top-ranked model
(cdk2x2_512.cif) of every seed is scored with perf/b2z2_fusebias/score.py's `native`, the instrument that
measured unfused silu's -0.05/-0.07 CA-lDDT on Protenix-v2, and the arms are paired by seed.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
from score import GT, native  # noqa: E402
from of3_score_ref import ca_map  # noqa: E402


def arm(run: Path, gt):
    out = {}
    for d in sorted(run.glob("struct_cdk2x2_512_s*")):
        out[int(d.name.rsplit("_s", 1)[1])] = native(d / "cdk2x2_512.cif", 512, gt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    gt = ca_map(GT)
    B, T = arm(a.base, gt), arm(a.test, gt)
    seeds = sorted(set(B) & set(T))
    rep = {"base": str(a.base), "test": str(a.test), "seeds": seeds, "domains": {}}
    for dom in B[seeds[0]]:
        for k in ("lddt_ca", "ca_rmsd_A"):
            b = np.array([B[s][dom][k] for s in seeds])
            t = np.array([T[s][dom][k] for s in seeds])
            rep["domains"].setdefault(dom, {})[k] = {
                "base": b.round(5).tolist(), "test": t.round(5).tolist(),
                "paired_mean_delta": round(float((t - b).mean()), 5),
                "base_spread": round(float(b.max() - b.min()), 5),
                "worst_base_minus_best_test": round(float(b.min() - t.max()), 5)}
    js = json.dumps(rep, indent=1)
    print(js)
    if a.out:
        a.out.write_text(js + "\n")


if __name__ == "__main__":
    main()
