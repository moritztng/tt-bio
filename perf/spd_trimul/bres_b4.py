#!/usr/bin/env python3
"""Falsifier for a --fast lever: the triangle contraction (bmm_res) reading a and b as bfloat4_b instead of bfp8.

bmm_res moves every byte of a, b and out once, so at RFD4's shape (hidden 256, 352 tokens, bfp8 a/b and bfp8 out under
trimul_t1b8 + pair_x8) it moves ~101 MB a call; bfp4 a/b cut that to ~67 MB. This times both arms back to back on one
chip (slope of `calls` enqueued calls over one sync) and grades each against a float64 einsum of the unrounded
inputs, so the bfp4 arm's error is read beside the bfp8 arm's.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=<tree> python perf/spd_trimul/bres_b4.py --n 352 --out <file>.jsonl

The lever is worth building into T1 only if bfp4 saves >= ~0.2 ms a call at 352 untimed (2 % of a 143 ms pass over
14 contractions) at an error the fast grade can absorb.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--calls", type=int, default=14)
ap.add_argument("--out", required=True)
a_ = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import bmm_res as BR  # noqa: E402

dev = T.get_device(trace=None)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
out = open(a_.out, "a")
for n in [int(s) for s in a_.ns.split(",")]:
    g = torch.Generator().manual_seed(n)
    # trimul a/b are p * sigmoid(g) projections of a layer-normed z: a gated Gaussian is the proxy
    A = torch.randn(1, a_.c, n, n, generator=g) * torch.sigmoid(torch.randn(1, a_.c, n, n, generator=g))
    B = torch.randn(1, a_.c, n, n, generator=g) * torch.sigmoid(torch.randn(1, a_.c, n, n, generator=g))
    for ta, tb in ((False, True), (True, False)):
        ref = torch.matmul(A.double().transpose(-1, -2) if ta else A.double(),
                           B.double().transpose(-1, -2) if tb else B.double())
        arms = {}
        for name, dt in (("b8", ttnn.bfloat8_b), ("b4", ttnn.bfloat4_b)):
            a = ttnn.from_torch(A, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            b = ttnn.from_torch(B, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            f = lambda: BR.matmul(a, b, compute_kernel_config=ckc, transpose_a=ta, transpose_b=tb,  # noqa: E731
                                  out_dtype=ttnn.bfloat8_b)
            x = f()
            if x is None:
                arms[name] = None
                continue
            y = ttnn.to_torch(x).double()
            rel = float((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            arms[name] = {"rel_rms": rel, "a": a, "b": b, "f": f, "ms": []}
        for _ in range(a_.reps):          # interleave arms rep by rep
            for name, arm in arms.items():
                if arm is None:
                    continue
                xs = []
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(a_.calls):
                    xs.append(arm["f"]())
                ttnn.synchronize_device(dev)
                arm["ms"].append(1e3 * (time.perf_counter() - t) / a_.calls)
                for x in xs:
                    ttnn.deallocate(x)
        for name, arm in arms.items():
            row = {"n": n, "c": a_.c, "ta": ta, "tb": tb, "arm": name}
            row.update({"declined": True} if arm is None else
                       {"ms_med": round(statistics.median(arm["ms"]), 4), "ms": [round(v, 4) for v in arm["ms"]],
                        "rel_rms": round(arm["rel_rms"], 5)})
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")
        for arm in arms.values():
            if arm:
                ttnn.deallocate(arm["a"]); ttnn.deallocate(arm["b"])
out.close()
