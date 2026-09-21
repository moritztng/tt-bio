#!/usr/bin/env python3
"""Is the residual a square-weight transpose in the INSTRUMENT rather than in the port?

`device_gradient.py` transposes the device gradient back to checkpoint orientation with

    if tuple(gt.shape) != want and tuple(gt.shape)[::-1] == want: gt = gt.t()

`_w_tt` stores every weight on the card as `w.t()`, so the taped dW rule returns `dW^T`.
For a NON-square weight the shape test fires and the orientation is restored. For a SQUARE
weight the shapes are equal either way, the test never fires, and the comparison scores
`dW^T` against `dW`.

The prediction that separates an instrument defect from a port defect: a transposed
comparison of a matrix with no particular symmetry reads norm ratio ~1 (a transpose is
norm-preserving, exactly), cosine ~0, and therefore rel_l2 ~sqrt(2) ~ 1.4142 -- and it
reads that on EVERY square weight and on NO non-square one, in every arm, regardless of
what the arm does to the arithmetic. This script tests that against the shapes.
"""
import argparse
import json
import os
import sys

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

import torch


def load(path):
    d = json.load(open(path))
    rows = d["per_tensor"] if isinstance(d, dict) else d
    return {r["param"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-boundary",
                    default=os.path.join(refpath.DIFFCAP, "sub_boundary.pt"))
    ap.add_argument("--arms", nargs="+", required=True, help="label=sidecar.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    S = torch.load(a.sub_boundary, map_location="cpu", weights_only=False)
    ref = S["grad_f64"]
    shape = {f"diffusion_module.{k}": tuple(v.shape) for k, v in ref.items() if v is not None}

    arms = [(lab, load(p)) for lab, _, p in (s.partition("=") for s in a.arms)]
    base = arms[0][1]
    sq2d = [p for p in base if len(shape.get(p, ())) == 2 and shape[p][0] == shape[p][1]]
    other = [p for p in base if p not in set(sq2d)]

    def stats(names, m):
        got = [m[p] for p in names if p in m]
        if not got:
            return None
        rels = sorted(r["rel_l2"] for r in got)
        coss = sorted(r["cos"] for r in got)
        rr = sorted(r["r"] for r in got)
        sq = sum(r["diff_norm"] ** 2 for r in got)
        return {"n": len(got),
                "median_rel": rels[len(rels) // 2],
                "min_rel": rels[0], "max_rel": rels[-1],
                "median_cos": coss[len(coss) // 2],
                "min_cos": coss[0], "max_cos": coss[-1],
                "median_r": rr[len(rr) // 2],
                "mass_pct": round(sum(r["pct_of_model_mass"] for r in got), 4),
                "sum_diff_sq": sq}

    rep = {"what": __doc__.strip().splitlines()[0],
           "n_square_2d": len(sq2d), "n_other": len(other),
           "square_params": sorted(sq2d), "arms": {}}
    print(f"{len(sq2d)} square 2-D weights, {len(other)} others, of {len(base)} compared\n")
    for lab, m in arms:
        s, o = stats(sq2d, m), stats(other, m)
        tot = s["sum_diff_sq"] + o["sum_diff_sq"]
        rep["arms"][lab] = {"square": s, "other": o,
                            "square_share_of_squared_error_pct": round(100 * s["sum_diff_sq"] / tot, 4)}
        print(f"--- {lab}")
        for nm, g in (("square", s), ("other", o)):
            print(f"  {nm:<7} n={g['n']:<4} mass={g['mass_pct']:>8.4f}%  "
                  f"rel med={g['median_rel']:.4e} [{g['min_rel']:.3e},{g['max_rel']:.3e}]  "
                  f"cos med={g['median_cos']:+.4f} [{g['min_cos']:+.3f},{g['max_cos']:+.3f}]  "
                  f"r med={g['median_r']:.4f}")
        print(f"  square tensors are {rep['arms'][lab]['square_share_of_squared_error_pct']:.2f} %"
              f" of this arm's squared error\n")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
