#!/usr/bin/env python3
"""Where the template pair stack's device distance lives: masked region, real region, per block.

`grade.py` reads the whole [n, n, 64] pair and gets a device distance from float64 of 0.0588
against BindCraft 2's own 0.0018. `bcx-extramsa` hit the same shape of number on the extra-MSA
stack and most of it turned out to be the masked rows and columns, which nothing downstream
reads. This splits the same three arms by mask region on `grade.py`'s own capture, adds the
per-block distance so a divergence that grows block by block is visible, and adds the two arms
that say whether the device's distance is its bf16 STORAGE or its arithmetic:

  device        `splice.TemplatePairStackOnDevice._forward`, the swap
  f64           `tt_bio.af2_reference`'s template pair stack in float64
  host bf16     the same reference at bfloat16
  host bf16-in  the same reference in float64 on an input rounded to bfloat16 first, which is
                the rounding the device cannot avoid and the arithmetic it does not do

No device time is spent on anything the swap does not itself do: the pair handed to the card is
the pair `grade.py` captured from BindCraft 2's own round.
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_round")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from splice import TemplatePairStackOnDevice                           # noqa: E402


def cmp(ref, x, keep=None):
    a = np.asarray(ref, np.float64)
    b = np.asarray(x, np.float64)
    if keep is not None:
        a, b = a[keep], b[keep]
    a, b = a.ravel(), b.ravel()
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
            "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)),
            "max_abs": float(np.abs(a - b).max()), "norm_ref": float(np.linalg.norm(a)),
            "norm_ratio": float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-300))}


def host_stack(model, act, mask, dtype, blocks=None):
    """The reference's own blocks, one at a time, so each block's output can be read."""
    z = torch.from_numpy(np.asarray(act, np.float64).copy()).to(dtype)
    m = torch.from_numpy(np.asarray(mask, np.float64).copy()).to(dtype)
    outs = []
    with torch.no_grad():
        for blk in (blocks if blocks is not None else model.template.pair_stack):
            z = blk(z, m)
            outs.append(z.double().numpy())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = dict(np.load(args.capture))
    act, mask = np.squeeze(cap["act_in"]), np.squeeze(cap["pair_mask"])
    jax_out = np.squeeze(cap["act_out"])
    real = mask != 0
    rep = {"capture": args.capture, "n": int(act.shape[0]),
           "real_pairs": int(real.sum()), "masked_pairs": int((~real).sum()),
           "masked_fraction": float((~real).mean()), "loadavg_start": os.getloadavg()}

    import afgrad as A
    import meter as M
    dm, ref = A.load_models(A.DEFAULT_PARAMS, template=True)
    dev = A.Dev(dm.to_device())
    stack = TemplatePairStackOnDevice(dev, k_template=2)
    clock = M.Clock(1.0)
    clock.start()
    import time
    t0 = time.time()
    d_out = stack._forward(act, mask)
    t1 = time.time()
    clock.stop()
    rep["device"] = {"forward_s": round(t1 - t0, 3), "aiclk_during": clock.window(t0, t1)}

    f64 = host_stack(ref["f64"], act, mask, torch.float64)
    bf16 = host_stack(ref["bf16"], act, mask, torch.bfloat16)
    act_bf16 = torch.from_numpy(act).to(torch.bfloat16).double().numpy()
    bf16_in = host_stack(ref["f64"], act_bf16, mask, torch.float64)

    arms = {"device": d_out, "bc2_jax": jax_out, "host_bf16": bf16[-1],
            "host_f64_bf16_input": bf16_in[-1]}
    rep["whole_pair"] = {k: cmp(f64[-1], v) for k, v in arms.items()}
    rep["real_region"] = {k: cmp(f64[-1], v, real) for k, v in arms.items()}
    rep["masked_region"] = {k: cmp(f64[-1], v, ~real) for k, v in arms.items()}
    rep["f64_masked_region_norm"] = float(np.linalg.norm(f64[-1][~real]))
    rep["per_block_vs_f64"] = {
        "host_bf16": [cmp(f64[i], bf16[i], real)["rel_l2"] for i in range(len(f64))],
        "host_f64_bf16_input": [cmp(f64[i], bf16_in[i], real)["rel_l2"]
                                for i in range(len(f64))]}
    for region in ("whole_pair", "real_region"):
        rep[region]["device_over_bc2_jax"] = (rep[region]["device"]["rel_l2"]
                                              / max(rep[region]["bc2_jax"]["rel_l2"], 1e-300))
        rep[region]["device_over_host_bf16"] = (rep[region]["device"]["rel_l2"]
                                                / max(rep[region]["host_bf16"]["rel_l2"], 1e-300))
    rep["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))
    rep["loadavg_end"] = os.getloadavg()
    (out / "diag.json").write_text(json.dumps(rep, indent=1, default=str))
    for k in ("whole_pair", "real_region", "masked_region"):
        print(k, json.dumps({a: round(c["rel_l2"], 6) if isinstance(c, dict) else round(c, 3)
                             for a, c in rep[k].items()}), flush=True)
    print("per block", json.dumps(rep["per_block_vs_f64"]), flush=True)


if __name__ == "__main__":
    main()
