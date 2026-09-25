#!/usr/bin/env python3
"""of3t-apbleaf: A/A on the CAPTURE itself, not on the file bytes.

`FLOOR_AA_N384.json` compares the two arms' 2,736 parameter gradients. `mech.py` consumes the
captured OPERANDS, so those need their own comparison: two pickles of the same tensors need not
be byte-equal, and a byte difference in a pickle is not a numeric difference.
"""
import json, sys, torch
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["sites"]
B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["sites"]
da = {s["gamma_path"]: s for s in A}
db = {s["gamma_path"]: s for s in B}
keys = sorted(set(da) & set(db))
fields = ("x", "g", "gamma", "dW_device", "db_device", "xhat_device")
moved, worst, wk = 0, 0.0, None
for k in keys:
    for f in fields:
        u, v = da[k][f], db[k][f]
        if u is None or v is None:
            continue
        u = u.to(torch.float64).reshape(-1); v = v.to(torch.float64).reshape(-1)
        if not torch.equal(u, v):
            moved += 1
            r = float((u - v).norm() / (u.norm() + 1e-300))
            if r > worst:
                worst, wk = r, k + ":" + f
R = {"what": __doc__.strip().splitlines()[0], "a": sys.argv[1], "b": sys.argv[2],
     "sites_compared": len(keys), "fields_per_site": len(fields),
     "tensors_compared": len(keys) * len(fields), "tensors_that_moved": moved,
     "bit_identical": moved == 0, "worst_relative_move": worst, "worst": wk}
json.dump(R, open(sys.argv[3], "w"), indent=2)
print(json.dumps(R))
