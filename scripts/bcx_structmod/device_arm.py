"""Run the structure module on the card and grade every tensor against the float64 reference.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-structmod \
      python3 scripts/bcx_structmod/device_arm.py --ref perf/bcx_structmod/out/ref_n64.npz

The reference is `scripts/bcx_structmod/ref_float64.py`'s dump of BindCraft 2's own module under
`jax_enable_x64`. Nothing here is graded against another device arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import ttnn

from tt_bio import af2_structure as sm


def grade(name, got, want):
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    want = np.asarray(want, dtype=np.float64).reshape(-1)
    if got.shape != want.shape:
        raise SystemExit(f"{name}: device {got.shape} vs reference {want.shape}")
    dn, wn = np.linalg.norm(got), np.linalg.norm(want)
    cos = float(got @ want / (dn * wn)) if dn and wn else float("nan")
    rel = float(np.linalg.norm(got - want) / wn) if wn else float("nan")
    amax = float(np.max(np.abs(got - want)))
    return {"cos": cos, "rel_l2": rel, "max_abs": amax,
            "norm_device": float(dn), "norm_ref": float(wn)}


def to_np(t):
    return ttnn.to_torch(t).float().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--point-dtype", default="float32", choices=("bfloat16", "float32"))
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--num-layer", type=int, default=sm.NUM_LAYER)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None,
                    help="npz of the device tensors themselves, for an error census the "
                         "scalar grades cannot show")
    args = ap.parse_args()

    ref = np.load(args.ref)
    n = int(ref["single"].shape[0])
    raw = np.load(args.params)
    params = {k: raw[k] for k in raw.files if "structure_module" in k}

    dtypes = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    device = ttnn.open_device(device_id=args.device_id)
    try:
        weights = sm.StructureWeights(params, device,
                                      dtype=dtypes[args.dtype],
                                      point_dtype=dtypes[args.point_dtype])
        module = sm.AF2MultimerStructureModule(weights, num_layer=args.num_layer)

        def up(array, dtype):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(array, np.float32)),
                                   layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

        single = up(np.asarray(ref["single"])[None, None], dtypes[args.dtype])
        pair = up(np.asarray(ref["pair"])[None], dtypes[args.dtype])
        seq_mask = up(np.asarray(ref["seq_mask"]).reshape(1, 1, n, 1),
                      dtypes[args.point_dtype])

        t0 = time.time()
        act, traj, angles = module(single, pair, seq_mask)
        ttnn.synchronize_device(device)
        elapsed = time.time() - t0

        got_act = to_np(act).reshape(n, sm.C_S)
        got_traj = np.stack([np.concatenate([to_np(c).reshape(n, 1) for c in layer], axis=1)
                             .reshape(n, 3, 4) for layer in traj])
        got_ang = np.stack([to_np(a).reshape(n, 14) for a in angles])
    finally:
        ttnn.close_device(device)

    report = {
        "n": n, "dtype": args.dtype, "point_dtype": args.point_dtype,
        "forward_wall_s": elapsed, "ref": args.ref,
        "grades": {
            "act": grade("act", got_act, ref["act"]),
            "traj": grade("traj", got_traj, ref["traj"]),
            "traj_last": grade("traj_last", got_traj[-1], ref["traj"][-1]),
            "unnormalized_angles": grade(
                "unnormalized_angles", got_ang,
                np.asarray(ref["sidechains_unnormalized"]).reshape(-1, n, 14)),
        },
    }
    for layer in range(got_traj.shape[0]):
        report["grades"][f"traj_layer{layer}"] = grade(
            f"traj_layer{layer}", got_traj[layer], ref["traj"][layer])

    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        np.savez(args.dump, act=got_act, traj=got_traj, unnormalized_angles=got_ang)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
