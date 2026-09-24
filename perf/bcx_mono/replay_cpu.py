#!/usr/bin/env python3
"""Replay the captured Evoformer boundary through tt-bio's torch reference, masked and not.

`capture.json`: BindCraft 2 hands the 48 blocks a pair mask that is 0.0% zero on the complex,
19.3% zero on the target alone and 35.7% zero on the binder alone -- it buckets the token axis
to 32 and the complex happens to land on 192. `splice.py:249` passes only `masks["msa"]`, so the
device arm has run every padded fold with the pair track unmasked.

This tests that in float32 torch with no card in the loop: same input, same 48 reference blocks,
pair mask on and off, graded against BindCraft 2's own JAX output for the same recycle.

`masked_fold.json` settled the question on card before this finished -- 48 fp32 blocks on a host
at loadavg 75 is slower than the fold it is checking -- so this arm stands as the card-free
re-derivation of the same mechanism, on the reference class rather than on ttnn. It needs
`capture/*.npz`, which `capture.py` writes and the repo does not carry (25 MB).
"""
import argparse, json, pathlib, sys, time
import ml_dtypes
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from tt_bio.af2_reference import load_af2_model
from tt_bio.af2_weights import load_af2_state_dict

PARAMS = "/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz"


def f32(a):
    a = np.asarray(a)
    if a.dtype.kind == "V":          # savez round-trips bfloat16 as raw 2-byte void
        a = a.view(ml_dtypes.bfloat16)
    return a.astype(np.float32)


def t(a, dt=torch.float32):
    return torch.from_numpy(f32(a)).to(dt)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(model, msa, pair, msa_mask, pair_mask, k=48):
    for i in range(k):
        msa, pair = model.evoformer[i](msa, pair, msa_mask, pair_mask)
    return msa, pair


ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="*", default=["complex", "target_alone", "binder_alone"])
ap.add_argument("--blocks", type=int, default=48)
a = ap.parse_args()

state = load_af2_state_dict(PARAMS)
model = load_af2_model(state, template=False, trunk_dtype=torch.float32)
for p in model.parameters():
    p.requires_grad_(False)

out = {"params": PARAMS, "blocks": a.blocks, "cases": {}}
with torch.no_grad():
    for name in a.cases:
        z = np.load(HERE / "capture" / f"{name}.npz")
        msa, pair = t(z["in_msa"]), t(z["in_pair"])
        mm, pm = t(z["msa_mask"]), t(z["pair_mask"])
        ones = torch.ones_like(pm)
        ref_msa, ref_pair = z["out_msa"], z["out_pair"]
        res = {"n": int(pair.shape[0]),
               "pair_mask_zero_frac": float((pm == 0).float().mean()),
               "msa_mask_zero_frac": float((mm == 0).float().mean())}
        for arm, this_pm in (("masked", pm), ("pairmask_dropped", ones)):
            t0 = time.time()
            m_out, z_out = run(model, msa, pair, mm, this_pm, a.blocks)
            res[arm] = {
                "msa_rel_l2": round(rel(m_out.numpy(), f32(ref_msa)), 6),
                "pair_rel_l2": round(rel(z_out.numpy(), f32(ref_pair)), 6),
                "seconds": round(time.time() - t0, 1)}
            print(name, arm, res[arm], flush=True)
        out["cases"][name] = res

(HERE / "replay_cpu.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
