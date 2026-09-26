"""The structure module's VJP on the card, graded against float64.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-structmod \
      python3 scripts/bcx_structmod/device_vjp.py \
        --ref perf/bcx_structmod/out/ref_real_n288_fold.npz

The reference must be built with `--boundary fold`, which seeds the cotangent on act, traj and
the unnormalised angles -- the device module's own outputs. Seeding it on
`final_atom_positions` instead would put the host tail (`l2_normalize`, the torsion frames, the
atom14 build) inside the graded path, and none of that runs on the card, so the grade would be
measuring code this row did not write.

There is no hand-written backward anywhere in `tt_bio/af2_structure.py`. The forward runs under
`tt_bio.autograd.tape()` and the tape differentiates the same verbs the untaped forward calls,
which is the point: what is graded here is the shipped module, not a second implementation of
it.
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
from tt_bio import autograd as ag
from tt_bio import taped_ttnn


def grade(got, want, live=None, axis=0):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        raise SystemExit(f"device {got.shape} vs reference {want.shape}")
    if live is not None:
        idx = [slice(None)] * got.ndim
        idx[axis] = live
        got, want = got[tuple(idx)], want[tuple(idx)]
    g, w = got.ravel(), want.ravel()
    gn, wn = np.linalg.norm(g), np.linalg.norm(w)
    return {"cos": float(g @ w / (gn * wn)) if gn and wn else float("nan"),
            "rel_l2": float(np.linalg.norm(g - w) / wn) if wn else float("nan"),
            "max_abs": float(np.max(np.abs(g - w))),
            "norm_device": float(gn), "norm_ref": float(wn)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    ap.add_argument("--point-dtype", default="float32", choices=("bfloat16", "float32"))
    ap.add_argument("--num-layer", type=int, default=sm.NUM_LAYER)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    ref = np.load(args.ref)
    if "ct_traj" not in ref.files:
        raise SystemExit(f"{args.ref} was not built with --boundary fold")
    n = int(ref["single"].shape[0])
    live = np.asarray(ref["seq_mask"]) > 0
    raw = np.load(args.params)
    params = {k: raw[k] for k in raw.files if "structure_module" in k}
    dtypes = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}

    device = ttnn.open_device(device_id=args.device_id)
    try:
        weights = sm.StructureWeights(params, device, dtype=dtypes[args.dtype],
                                      point_dtype=dtypes[args.point_dtype])
        module = sm.AF2MultimerStructureModule(weights, num_layer=args.num_layer)

        def up(array, dtype):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(array, np.float32)),
                                   layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

        single_v = up(np.asarray(ref["single"])[None, None], dtypes[args.dtype])
        pair_v = up(np.asarray(ref["pair"])[None], dtypes[args.dtype])
        seq_mask = up(np.asarray(ref["seq_mask"]).reshape(1, 1, n, 1),
                      dtypes[args.point_dtype])

        single = ag.Tensor(single_v, requires_grad=True)
        pair = ag.Tensor(pair_v, requires_grad=True)

        t0 = time.time()
        with taped_ttnn.tape():
            act, traj, angles = module(single, pair, seq_mask)
        ttnn.synchronize_device(device)
        t_fwd = time.time() - t0

        ct_act = np.asarray(ref["ct_act"])
        ct_traj = np.asarray(ref["ct_traj"])                 # [L, n, 3, 4]
        ct_unnorm = np.asarray(ref["ct_unnormalized"])       # [L, n, 7, 2]

        roots, seeds = [act], [up(ct_act[None, None], act.value.dtype)]
        for layer in range(args.num_layer):
            flat = ct_traj[layer].reshape(n, 12)             # row-major, matching to_array
            for slot, component in enumerate(traj[layer]):
                roots.append(component)
                seeds.append(up(flat[:, slot].reshape(1, 1, n, 1), component.value.dtype))
            roots.append(angles[layer])
            seeds.append(up(ct_unnorm[layer].reshape(1, 1, n, 14), angles[layer].value.dtype))

        t1 = time.time()
        ag.backward(roots, seeds)
        ttnn.synchronize_device(device)
        t_bwd = time.time() - t1

        def down(t, shape):
            if t.grad is None:
                raise SystemExit("the tape produced no gradient for an input it was told "
                                 "requires one")
            return ttnn.to_torch(t.grad).float().numpy().reshape(shape)

        g_single = down(single, (n, sm.C_S))
        g_pair = down(pair, (n, n, sm.C_Z))
    finally:
        ttnn.close_device(device)

    report = {
        "n": n, "n_live": int(live.sum()), "dtype": args.dtype,
        "point_dtype": args.point_dtype, "num_layer": args.num_layer,
        "forward_wall_s": t_fwd, "backward_wall_s": t_bwd, "ref": args.ref,
        "grades_live": {"g_single": grade(g_single, ref["g_single"], live, 0),
                        "g_pair_rows": grade(g_pair, ref["g_pair"], live, 0)},
        "grades_all": {"g_single": grade(g_single, ref["g_single"]),
                       "g_pair": grade(g_pair, ref["g_pair"])},
    }
    gp, rp = np.asarray(g_pair, np.float64), np.asarray(ref["g_pair"], np.float64)
    blk = np.ix_(np.flatnonzero(live), np.flatnonzero(live))
    report["grades_live"]["g_pair_block"] = grade(gp[blk], rp[blk])

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    if args.dump:
        np.savez(args.dump, g_single=g_single, g_pair=g_pair)
    return 0


if __name__ == "__main__":
    sys.exit(main())
