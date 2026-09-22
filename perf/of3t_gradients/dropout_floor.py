#!/usr/bin/env python3
"""What varies in BUNDLE-MIN's gradient when nothing but the dropout mask varies.

`capture_trunk_boundary.py` replayed the bundle's own step with upstream's own code, upstream's
own checkpoint, `num_recycles` pinned to 0 and all 45 `torch.randn` draws and both
`random.random` values replayed from `draws_recycles0.pt` with zero shape mismatches -- and the
block gradients came back 1.40 to 1.92 relative against the published ones, on a 5.0e-02 bar.
Upstream failing to reproduce upstream is either a defect in the replay or an input nobody
recorded. This says which.

OF3's pair stack applies `DropoutRowwise`/`DropoutColumnwise` at r = 0.25 in train mode
(`primitives/dropout.py:58`, `mask = x.new_ones(shape); mask = self.dropout(mask)`). That draw
comes from the default generator OF THE TENSOR'S DEVICE. The bundle's step ran on an A100, so
its dropout masks came from the CUDA generator; a replay on this CPU draws its own, and
`draws_recycles0.pt` records the recycle count, the diffusion noise, `torch.randn` and
`random.random` -- not these. Restoring an RNG state reproduces a state, not a value.

The measurement below needs no model forward. It takes the captured block boundary, which fixes
every other input exactly, and differentiates the SAME block under several dropout draws:

  * dropout ON, several seeds: the spread is the floor any consumer of this bundle is measuring
    against on a different device;
  * dropout OFF, twice: the control. If the spread survives with dropout off, the cause is
    something else and this whole reading is wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "perf/of3t_gradients"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BUNDLE = "/home/ttuser/of3t/bundle_min"
CAP = "/tmp/of3t/of3t-gradients/cap"
PER_TENSOR_BAR = 5.0e-02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()

    import numpy as np
    import torch
    from openfold3.core.model.primitives.dropout import Dropout
    from instrument_a_stack import their_stack, load_ckpt, rel_l2

    i = a.block
    seeds = [int(x) for x in a.seeds.split(",")]
    cap = torch.load(os.path.join(CAP, f"block{i}_boundary.pt"),
                     map_location="cpu", weights_only=False)
    args_in = cap["args"]
    s_in, z_in = args_in[0].to(torch.float64), args_in[1].to(torch.float64)
    kw = cap.get("kwargs") or {}
    sm = (args_in[2] if len(args_in) > 2 else kw["single_mask"]).to(torch.float64)
    pm = (args_in[3] if len(args_in) > 3 else kw["pair_mask"]).to(torch.float64)
    cot_s, cot_z = cap["cot"][0].to(torch.float64), cap["cot"][1].to(torch.float64)

    sd = load_ckpt()
    mods, dims, _ = their_stack(sd, 1, first=i)
    blk = mods[0]
    n_drop = sum(1 for m in blk.modules() if isinstance(m, Dropout))
    rates = sorted({float(m.r) for m in blk.modules() if isinstance(m, Dropout)})
    rep = {"instrument": "the dropout floor of BUNDLE-MIN's per-parameter gradient",
           "block": i, "tokens": int(z_in.shape[1]),
           "dropout_modules_in_block": n_drop, "dropout_rates": rates,
           "bar": PER_TENSOR_BAR,
           "probe": "the captured block boundary of the bundle's own step; every input but the "
                    "dropout mask is fixed exactly"}
    print(f"block {i}: {n_drop} Dropout modules, rates {rates}", flush=True)

    def grads_under(seed, train):
        blk.train(train)
        for p in blk.parameters():
            p.grad = None
        torch.manual_seed(seed)
        s, z = blk(s_in, z_in, sm, pm)
        ((s * cot_s).sum() + (z * cot_z).sum()).backward()
        return {n: p.grad.detach().clone() for n, p in blk.named_parameters()}

    runs = {}
    for s_ in seeds:
        runs[f"train_seed{s_}"] = grads_under(s_, True)
        print(f"  dropout ON  seed {s_} done", flush=True)
    for s_ in seeds[:2]:
        runs[f"eval_seed{s_}"] = grads_under(s_, False)
        print(f"  dropout OFF seed {s_} done", flush=True)

    def spread(a_key, b_key):
        A, B = runs[a_key], runs[b_key]
        rows = [(rel_l2(A[n].numpy(), B[n].numpy()), n) for n in A]
        rows.sort(reverse=True)
        return {"worst": rows[0][0], "worst_tensor": f"pairformer_stack.blocks.{i}.{rows[0][1]}",
                "median": float(np.median([r for r, _ in rows])), "n": len(rows),
                "over_bar": sum(1 for r, _ in rows if r > PER_TENSOR_BAR)}

    rep["dropout_on_pairs"] = {
        f"{seeds[k]}_vs_{seeds[k+1]}": spread(f"train_seed{seeds[k]}", f"train_seed{seeds[k+1]}")
        for k in range(len(seeds) - 1)}
    rep["dropout_off_control"] = spread(f"eval_seed{seeds[0]}", f"eval_seed{seeds[1]}")

    # And against the published entry itself, so the floor and the disagreement are on one axis.
    ref_all = torch.load(os.path.join(BUNDLE, "grads_f64_recycles0.pt"),
                         map_location="cpu", weights_only=False)
    pre = f"pairformer_stack.blocks.{i}."
    g_ref = {k[len(pre):]: v for k, v in ref_all.items() if k.startswith(pre)}
    del ref_all
    for key in (f"train_seed{seeds[0]}", f"eval_seed{seeds[0]}"):
        rows = [(rel_l2(runs[key][n].numpy(), g_ref[n].to(torch.float64).numpy()), n)
                for n in runs[key] if g_ref.get(n) is not None]
        rows.sort(reverse=True)
        rep[f"vs_bundle_{key}"] = {
            "worst": rows[0][0], "worst_tensor": pre + rows[0][1],
            "median": float(np.median([r for r, _ in rows])), "n": len(rows),
            "over_bar": sum(1 for r, _ in rows if r > PER_TENSOR_BAR)}

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"dropout_floor_block{i}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "probe"}, indent=1, default=str))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
