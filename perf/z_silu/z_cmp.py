#!/usr/bin/env python3
"""Compare two z-silu arm dumps against each other and against a torch fp32 reference."""
import json, sys
import numpy as np, torch

SHAPES = {"298": (1, 30, 298, 256), "512": (1, 16, 512, 256)}
N_OUT = 1024


def stats(x, y):
    d = (x - y).ravel()
    rms = float(np.sqrt((d ** 2).mean()))
    denom = float(np.sqrt((y.ravel() ** 2).mean()))
    xc, yc = x.ravel() - x.mean(), y.ravel() - y.mean()
    pcc = float((xc * yc).sum() / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return dict(equal=bool(np.array_equal(x, y)), max_abs=float(np.abs(d).max()),
                rmsd=rms, rel_rmsd=rms / denom, pcc=pcc)


shape = sys.argv[1]
paths = sys.argv[2:]
arms = {p.split("/")[-1].replace(".npy", ""): np.load(p) for p in paths}

shp = SHAPES[shape]
torch.manual_seed(0)
ta = torch.randn(*shp).to(torch.bfloat16).float()
tb = torch.randn(256, N_OUT).to(torch.bfloat16).float()
ref = torch.nn.functional.silu(ta @ tb).numpy()          # fp32 reference on the bf16 inputs

res = {}
names = list(arms)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        res[f"{names[i]} vs {names[j]}"] = stats(arms[names[i]], arms[names[j]])
for n in names:
    res[f"{n} vs torch_fp32"] = stats(arms[n], ref)
print(json.dumps(res, indent=1))
