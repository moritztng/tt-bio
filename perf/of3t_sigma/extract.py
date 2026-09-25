#!/usr/bin/env python3
"""of3t-sigma step 1a: per draw, carry the device dump into upstream names with score.py's own
loader and keep ours / float64 / upstream-bf16 on the SCORED set in one float64 file per draw.

    extract.py --k 3 --out /home/ttuser/of3t_sigma/scored_s3.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
import score  # noqa: E402

S, P = Path("/home/ttuser/of3t_paedraws"), HERE.parent / "of3t_paedraws"

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T = f"PD384_s{a.k}"
ref = {k: v.to(torch.float64) for k, v in
       torch.load(S / f"ref_s{a.k}/f64/grads_f64.pt", weights_only=False).items()
       if v is not None and bool(v.any())}
bf = {k: v.to(torch.float64) for k, v in
      torch.load(S / f"ref_s{a.k}/bf16/grads_bf16.pt", weights_only=False).items() if v is not None}
bij = json.loads((P / f"BIJECTION_{T}.json").read_text())
shapes = json.loads((P / f"DEVICE_SHAPES_{T}.json").read_text())["shapes"]
ours, _ = score.to_upstream(score.load_device(S / f"grad_{T}.pt", shapes), bij)
scored = sorted(set(ref) & (set(bij["placements"]) | set(bij["derived"])))
out = {"f64": {k: ref[k] for k in scored},
       "ours": {k: ours.get(k, torch.zeros_like(ref[k])) for k in scored},
       "bf16": {k: bf.get(k, torch.zeros_like(ref[k])) for k in scored}}
torch.save(out, a.out)
print(T, len(scored), "tensors ->", a.out)
