#!/usr/bin/env python3
"""Per-tensor rel (ours, bf16, each against float64 at the same draw) and float64 mass, for the
six A44 draws, with score.py's own loaders. Read-only on every input.

    pertensor_draws.py W OUT.json     W = a tree holding perf/of3t_{fullstep64,confpfe,paedraws}
"""
import json
import sys
from pathlib import Path

import torch

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(W / "perf/of3t_fullstep64"))
import score  # noqa: E402

S = Path("/home/ttuser/of3t_paedraws")
C = Path("/home/ttuser/of3t_confpfe")


def paths(k):
    if k == 0:
        return (C / "ref384c/f64/grads_f64.pt", C / "ref384c/bf16/grads_bf16.pt",
                C / "grad_CF384.pt", W / "perf/of3t_confpfe", "CF384")
    return (S / f"ref_s{k}/f64/grads_f64.pt", S / f"ref_s{k}/bf16/grads_bf16.pt",
            S / f"grad_PD384_s{k}.pt", W / "perf/of3t_paedraws", f"PD384_s{k}")


out = {}
for k in range(6):
    f64, b16, dev, P, T = paths(k)
    ref = {n: v.to(torch.float64) for n, v in torch.load(f64, weights_only=False).items()
           if v is not None and bool(v.any())}
    bf = {n: v.to(torch.float64) for n, v in torch.load(b16, weights_only=False).items()
          if v is not None}
    bij = json.loads((P / f"BIJECTION_{T}.json").read_text())
    shapes = json.loads((P / f"DEVICE_SHAPES_{T}.json").read_text())["shapes"]
    arm, _ = score.to_upstream(score.load_device(dev, shapes), bij)
    scored = sorted(set(ref) & (set(bij["placements"]) | set(bij["derived"])))
    d = {}
    for n in scored:
        m = float((ref[n] ** 2).sum())
        rel = lambda x: (float(((x[n] - ref[n]) ** 2).sum()) / m) ** 0.5 if n in x else 1.0
        d[n] = [m, rel(arm), rel(bf)]
    out[k] = d
    print(k, T, len(d), flush=True)
    del ref, bf, arm
json.dump(out, open(OUT, "w"))
