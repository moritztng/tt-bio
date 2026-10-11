"""Wrong-value rate of the transition's matmuls on Wormhole per K block, layout and throttle level.

    TT_VISIBLE_DEVICES=N python perf/spd_trans2/erratum_rate.py --out OUT.json [--draws 300] [--arms a,b,...]

The fp32-dest erratum (state/spd-wherr.md) writes +-2 / +-4 into ~1e-8 of the outputs when more than one K tile
accumulates in dest: outlier_stage.py saw 2 in 139M for the sharded fc3 at K block 4, HiFi3. A rate that low needs
~1e9 outputs per arm, so the reference here is the same matmul at K block 1 ON DEVICE (clean in every audit) and
only the difference goes to the host: an output is bad when it is more than 0.25 off the reference. Each arm also
runs twice on the same input: `flaky` counts outputs that differ between the two runs, which a deterministic logic
fault cannot produce and a timing (di/dt) fault does. `throttle` arms set ttnn's MM_THROTTLE (NOPs between MVMULs,
level 1 = 73 % of peak), the knob ttnn ships for "di/dt problems on more than 48 cores" on Wormhole.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--draws", type=int, default=300)
ap.add_argument("--arms", default="")
ap.add_argument("--fid", default="hifi3")
a = ap.parse_args()

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
GX, GY = T.COMPUTE_GRID_MAIN
FID = {"hifi2": ttnn.MathFidelity.HiFi2, "hifi3": ttnn.MathFidelity.HiFi3, "hifi4": ttnn.MathFidelity.HiFi4}[a.fid]


def ckc(throttle=0):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=FID, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True,
        throttle_level=[ttnn.ThrottleLevel.NO_THROTTLE, ttnn.ThrottleLevel.LEVEL_1, ttnn.ThrottleLevel.LEVEL_2,
                        ttnn.ThrottleLevel.LEVEL_3][throttle])


def cdiv(x, y):
    return -(-x // y)


def subblock(h, w):
    return max(((sh, sw) for sh in range(1, 5) for sw in range(1, 5)
                if sh * sw <= 4 and h % sh == 0 and w % sw == 0), key=lambda t: (t[0] * t[1], t[1]))


def pc_1d(mt, nt, bw):
    pm = cdiv(mt, GX * GY)
    sh, sw = subblock(pm, nt)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=nt, per_core_M=pm, per_core_N=nt, fuse_batch=True, fused_activation=None,
        mcast_in0=False)


def pc_2d(mt, nt, bw, gx=8):
    pm, pn = cdiv(mt, GY), nt // gx
    sh, sw = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, cdiv(mt, pm)), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


# name -> (rows, K, N, layout, K block, throttle). pair = the c730 normal 5-row block, msa = its 16-row block.
ARMS = {}
for thr in (0, 1):
    t = f":t{thr}" if thr else ""
    for bw in (2, 4, 8):
        ARMS[f"pair.fc3:1d:bw{bw}{t}"] = (5 * 736, 1024, 256, "1d", bw, thr)
    ARMS[f"pair.fc3:2d:bw4{t}"] = (5 * 736, 1024, 256, "2d", 4, thr)
    ARMS[f"pair.fc1:2d:bw8{t}"] = (5 * 736, 256, 1024, "2d", 8, thr)
    ARMS[f"msa.fc3:1d:bw2{t}"] = (16 * 736, 512, 128, "1d", 2, thr)
    ARMS[f"msa.fc1:2d:bw4{t}"] = (16 * 736, 128, 512, "2d", 4, thr)
names = a.arms.split(",") if a.arms else list(ARMS)

res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "grid": [GX, GY],
       "fidelity": a.fid, "draws": a.draws, "arms": {}}
for name in names:
    M, K, N, lay, bw, thr = ARMS[name]
    mt, nt = M // 32, N // 32
    pc = pc_1d(mt, nt, bw) if lay == "1d" else pc_2d(mt, nt, bw)
    ref_pc = pc_1d(mt, nt, 1) if lay == "1d" else pc_2d(mt, nt, 1)
    kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bad = flaky = 0
    worst = 0.0
    t0 = time.monotonic()
    g = torch.Generator().manual_seed(7)
    try:
        for d in range(a.draws):
            xh = torch.randn(1, 1, M, K, generator=g) * torch.sigmoid(torch.randn(1, 1, M, K, generator=g))
            wh = torch.randn(K, N, generator=g) / K ** 0.5
            x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
            w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            outs = [ttnn.linear(x, w, program_config=p, compute_kernel_config=c, **kw)
                    for p, c in ((ref_pc, ckc(0)), (pc, ckc(thr)), (pc, ckc(thr)))]
            r, o1, o2 = (ttnn.to_torch(o).float() for o in outs)
            for t in outs + [x, w]:
                ttnn.deallocate(t)
            e = (o1 - r).abs()
            bad += int((e > 0.25).sum())
            worst = max(worst, float(e.max()))
            flaky += int((o1 != o2).sum())
    except Exception as e:  # noqa: BLE001
        res["arms"][name] = dict(err=str(e).splitlines()[0][:200])
        print(json.dumps({name: res["arms"][name]}), flush=True)
        continue
    res["arms"][name] = dict(outputs=M * N * a.draws, bad=bad, rate=bad / (M * N * a.draws), max_err=worst,
                             flaky=flaky, s=round(time.monotonic() - t0, 1))
    print(json.dumps({name: res["arms"][name]}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
