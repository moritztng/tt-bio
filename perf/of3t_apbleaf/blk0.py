#!/usr/bin/env python3
"""of3t-apbleaf: block 0's error mass is NOT in layer_norm_a, so where is it.

The brief names blocks 44, 4 and 0. `BYBLOCK_*_N384.json` shows layer_norm_a holds 90.79 % and
96.35 % of blocks 44 and 4 and 0.0177 % of block 0, so block 0 belongs to a different leaf and
this says which one. Handed to the orchestrator rather than taken: this row owns one leaf.
"""
import argparse, json, torch

PRE = "pairformer_stack.blocks."


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "grads" in d and isinstance(d["grads"], dict):
        d = d["grads"]
    return {(k if k.startswith(PRE) else PRE + k): v.to(torch.float64).reshape(-1)
            for k, v in d.items() if v is not None and hasattr(v, "shape")
            and (k.startswith(PRE) or k.split(".")[0].isdigit())}


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--block", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ref, ours = load(a.ref), load(a.ours)
keys = sorted(set(ref) & set(ours))
tot = sum(float((ours[k] - ref[k]).pow(2).sum()) for k in keys)
bk = [k for k in keys if int(k.split(".")[2]) == a.block]
rows = []
for k in bk:
    e = float((ours[k] - ref[k]).pow(2).sum())
    rn = float(ref[k].norm())
    rows.append({"param": k, "leaf": ".".join(k.split(".")[3:]), "err_sq": e,
                 "abs_err": e ** 0.5, "ref_norm": rn,
                 "share_of_trunk_error": e / tot,
                 "rel": (e ** 0.5) / rn if rn else None})
rows.sort(key=lambda r: -r["err_sq"])
json.dump({"what": __doc__.strip().splitlines()[0], "block": a.block,
           "total_error_mass_sq": tot, "rows": rows}, open(a.out, "w"), indent=2)
print("block %d, top leaves by share of the TRUNK's error mass:" % a.block)
for r in rows[:8]:
    print("  %-52s abs=%.6e refn=%.6e share=%8.4f%% rel=%.4f"
          % (r["leaf"], r["abs_err"], r["ref_norm"], 100 * r["share_of_trunk_error"], r["rel"]))
