"""The masked arm on the CARD against BindCraft 2's own JAX Evoformer, at the state its trajectories ran.

`perf/bcx_mask/pairmask_grade.py` graded host torch emulations of the device's masking. This runs
the shipped device blocks from the tree on PYTHONPATH: BindCraft 2's captured 48-block input and
masks (`sequence_gradients`, first recycle, bucket 32: n=211 with 19 pad at 77..95; bucket 1:
n=192, no pad) through `AF2DeviceModel.device_evoformer`, untaped, and scores msa / pair /
single on the real residues against the JAX output of the same call.

  masked    msa mask and `af2_pair_masks(pair_mask)`: what the candidate ships
  unmasked  masks None: the only thing `origin/main` can run, since it asserts the mask away

    PYTHONPATH=<tree> TT_VISIBLE_DEVICES=3 python3 perf/bcx_landmask/jax_grade.py --cap DIR --out x.json
"""
import argparse, json, os, sys, threading, time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cap", required=True, help="dir with capture_grad_b{1,32}.npz (bcx-mask's)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
tree = Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()
sys.path.insert(0, str(tree))
from perf.bcx_afgrad import afgrad as A
from perf.bcx_stack.stack import Clock
from tt_bio.af2 import af2_pair_masks

dm, ref = A.load_models(A.DEFAULT_PARAMS)
dev = A.Dev(dm)
single = ref["f32"].single_activations
clock = Clock()


def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"rel_l2": float((a - b).norm() / b.norm()), "cos": float(a @ b / (a.norm() * b.norm()))}


out = {"tree": str(tree), "host": os.uname().nodename, "aiclk_node": clock.path,
       "loadavg_start": open("/proc/loadavg").read().split()[:3], "buckets": {}}
for bucket in (32, 1):
    with np.load(Path(args.cap) / f"capture_grad_b{bucket}.npz") as z:
        cap = {k: torch.from_numpy(z[k]).float() for k in z.files}
    m0, z0, mm, pm = cap["msa_in"], cap["pair_in"], cap["msa_mask"], cap["pair_mask"]
    keep = torch.diagonal(pm).bool()
    n = int(pm.shape[0])
    rec = {"n": n, "n_masked": int((~keep).sum()), "arms": {}}
    with torch.no_grad():
        js = single(cap["msa_out"][0])
    arms = {"masked": (dev.up(mm), af2_pair_masks(pm, dev.device))}
    if bucket == 32:
        arms["unmasked"] = (None, (None, None))
    for name, (msa_mask, pair_masks) in arms.items():
        t0 = time.time()
        m, z = dev.stack(dev.up(m0), dev.up(z0), 0, 48, msa_mask=msa_mask, pair_masks=pair_masks)
        dev.sync()
        span = (t0, time.time())
        m, z = dev.down(m, m0.shape), dev.down(z, z0.shape)
        with torch.no_grad():
            s = single(m[0])
        rec["arms"][name] = {
            "msa": cmp(m[:, keep], cap["msa_out"][:, keep]),
            "pair": cmp(z[keep][:, keep], cap["pair_out"][keep][:, keep]),
            "single": cmp(s[keep], js[keep]),
            "finite": bool(torch.isfinite(m).all() and torch.isfinite(z).all()),
            "wall_s": span[1] - span[0], "aiclk": clock.window([span]),
            "load1": float(open("/proc/loadavg").read().split()[0])}
        print(bucket, name, {k: rec["arms"][name][k] for k in ("msa", "pair", "single", "aiclk")}, flush=True)
    out["buckets"][f"bucket_{bucket}"] = rec
clock.stop()
json.dump(out, open(args.out, "w"), indent=1)
