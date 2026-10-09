"""Wrong values in the pair Transition, interleaved path against transition_shard, per fidelity.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/outlier_ab.py --out OUT.json [--S 512,736] [--seeds 0,1]

transition_shard moved Protenix-v2's cdk2x2_512 CA-lDDT against 1HCL from 0.900 / 0.861 (Wormhole interleaved
path) to 0.937 / 0.914, back to the published level. Its rel_rms against float64 barely differs (0.00351 vs
0.00341), but rel_rms cannot see a few O(1) wrong values per call, the class spd-overhead pinned on Wormhole
HiFi4 + fp32 dest acc (fc2 writes -4.0 where float64 says 0.00039). This runs the real Transition module with
the lever off and on, at the fold's trunk fidelity and at HiFi4, and counts the output pixels whose error
against float64 exceeds --bad, with the worst error and rel_rms beside it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--S", default="512,736")
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--bad", type=float, default=0.1)
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = "normal"

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers("normal")
HIFI4 = CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True,
                packer_l1_acc=True)
CKCS = {"trunk": T.trunk_compute_kernel_config(HIFI4), "hifi4": HIFI4}
with T.levers("fast"):  # the trunk config fast mode builds: its fidelity and acc levers on the same base
    CKCS["fast"] = T.trunk_compute_kernel_config(HIFI4)
C, HID = 256, 1024
F = torch.nn.functional
bf = lambda t: t.to(torch.bfloat16).to(torch.float64)  # noqa: E731

res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH,
       "trunk_fidelity": str(CKCS["trunk"].math_fidelity), "bad": a.bad, "cells": []}
for seed in (int(s) for s in a.seeds.split(",")):
    for S in (int(s) for s in a.S.split(",")):
        g = torch.Generator().manual_seed(seed)
        sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
              "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
              "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
              "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
              "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
        zt = bf(torch.randn(1, S, S, C, generator=g))
        ref = torch.empty(1, S, S, C, dtype=torch.float64)
        for r in range(0, S, 64):  # float64 reference in row blocks: the full hidden is 8 GB at 736
            xn = F.layer_norm(zt[:, r:r + 64], (C,), sd["norm.weight"], sd["norm.bias"], 1e-5)
            ref[:, r:r + 64] = (F.silu(xn @ sd["fc1.weight"].T) * (xn @ sd["fc2.weight"].T)) @ sd["fc3.weight"].T
        z = ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for ck_name, ckc in CKCS.items():
            tr = T.Transition({k: v.float() for k, v in sd.items()}, ckc)
            mode = "fast" if ck_name == "fast" else "normal"
            for arm, levers in (("interleaved", f"{mode}-transition_shard"), ("shard", f"{mode}+transition_shard")):
                T.TRANSITION_H_CHUNK_SHAPES.clear()
                with T.levers(levers):
                    o = tr(z)
                out = ttnn.to_torch(o).to(torch.float64)
                ttnn.deallocate(o)
                err = (out - ref).abs()
                px = err.amax(-1)[0]
                cell = dict(seed=seed, S=S, ckc=ck_name, arm=arm, h=[k[2] for k in T.TRANSITION_H_CHUNK_SHAPES],
                            n_bad_px=int((px > a.bad).sum()), max_err=float(px.max()),
                            rel_rms=float((out - ref).norm() / ref.norm()),
                            mean_err=float((out - ref).mean()), worst_px=[int(i) for i in divmod(int(px.argmax()), S)])
                res["cells"].append(cell)
                print(json.dumps(cell), flush=True)
            del tr
        ttnn.deallocate(z)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
