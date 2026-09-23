#!/usr/bin/env python3
"""of3t-pwaslice: the fix may only ADD gradient to the leaves that had none.

    compare.py --base grad_B.pt --fix grad_F.pt --base-json DEV_B.json --fix-json DEV_F.json --out F.json

Every named gradient except the D262 weights must be bit-identical between base and fix on the
same card; the D262 weights must be zero (or absent) on base and nonzero on fix; the loss must be
bit-identical. id()-keyed `._wc.` aliases are excluded, as AA_T64R.json does: their names differ
per process.
"""
import argparse
import json
import re

import torch

D262 = re.compile(r"trunk\.msa_module\.blocks\.\d+\.pwa\.[mgo]_weight$")


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--base", "--fix", "--base-json", "--fix-json", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    b, f = torch.load(a.base, map_location="cpu"), torch.load(a.fix, map_location="cpu")
    keep = lambda d: {k: v for k, v in d.items() if "._wc." not in k}
    b, f = keep(b), keep(f)
    names = sorted(set(b) | set(f))
    target = [k for k in names if D262.search(k)]
    others = [k for k in names if not D262.search(k)]
    differ, missing = [], []
    for k in others:
        if k not in b or k not in f:
            missing.append(k)
        elif b[k].shape != f[k].shape or not torch.equal(b[k], f[k]):
            differ.append(k)
    nz = lambda d, k: k in d and bool(d[k].abs().max() > 0)
    lb, lf = json.load(open(a.base_json))["loss"], json.load(open(a.fix_json))["loss"]
    rec = {"base": a.base, "fix": a.fix, "n_named": len(names), "others": len(others),
           "others_bit_identical": len(others) - len(differ) - len(missing),
           "others_differ": differ[:20], "n_differ": len(differ),
           "only_in_one": missing[:20], "n_only_in_one": len(missing),
           "d262": {k: {"base_nonzero": nz(b, k), "fix_nonzero": nz(f, k),
                        "fix_norm": float(f[k].double().norm()) if k in f else None}
                    for k in target},
           "loss_base": lb, "loss_fix": lf, "loss_bit_identical": lb == lf,
           "note": "id()-keyed ._wc. aliases excluded (names differ per process)"}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("others", "others_bit_identical", "n_differ",
                                          "n_only_in_one", "loss_bit_identical")}
                     | {"d262_fix_nonzero": sum(v["fix_nonzero"] for v in rec["d262"].values()),
                        "d262_base_nonzero": sum(v["base_nonzero"] for v in rec["d262"].values()),
                        "d262_n": len(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
