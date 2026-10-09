"""Transition fc3 ([rows, hidden] x [hidden, c]) under every program config that fits: us per call and accuracy.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/fc3_bench.py --out OUT.json [--b8]

fc3 runs 114 us a call on WH against 61 us for fc1's matmul of the same FLOPs (census c730): with N = 8 tiles the
auto 2D config gives each core one output column and multicasts a 32-tile-deep in0 row block along the grid. Arms:
  auto     ttnn.linear(core_grid=CORE_GRID_MAIN), what the fold runs
  2d:...   MatmulMultiCoreReuseMultiCastProgramConfig, grid gx x gy, in0_block_w, subblock
  1d:...   MatmulMultiCoreReuseMultiCast1DProgramConfig with mcast_in0=False: every core owns per_core_M rows and
           the whole weight is multicast once
Input in L1 (as the fold's x_1), output bf16 to DRAM, Protenix's transition config (HiFi4, fp32 dest, packer L1
acc). Accuracy is rel_rms against float64 of the same bf16 operands, plus a digest: equal digests mean equal bytes.
--b8 runs the fast-mode dtypes (bfp8 hidden and weight).
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--batches", type=int, default=7)
ap.add_argument("--b8", action="store_true")
ap.add_argument("--shapes", default="pair,msa")
a = ap.parse_args()

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
samples = []


def _sample():
    row = {}
    for n in OPENED:
        try:
            v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            if 100 <= v <= 3000:
                row[n] = v
        except Exception:
            pass
    samples.append((time.monotonic(), row))


def _sampler():
    while True:
        _sample()
        time.sleep(0.2)


threading.Thread(target=_sampler, daemon=True).start()


def clock(t0, t1):
    _sample()
    t1 = max(t1, samples[-1][0])
    v = sorted(x for ts, r in samples if t0 <= ts <= t1 for x in r.values())
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


def sync():
    ttnn.synchronize_device(dev)


def timed(fn, batches):
    for _ in range(2):
        ttnn.deallocate(fn())
    sync()
    ts = []
    t0 = time.monotonic()
    for _ in range(batches):
        sync()
        t = time.perf_counter()
        for _ in range(10):
            ttnn.deallocate(fn())
        sync()
        ts.append((time.perf_counter() - t) / 10 * 1e6)
    return dict(us_min=round(min(ts), 2), us_med=round(st.median(ts), 2), clock=clock(t0, time.monotonic()))


CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
CKC = CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True,
              packer_l1_acc=True)
DT = ttnn.bfloat8_b if a.b8 else ttnn.bfloat16
GX, GY = T.COMPUTE_GRID_MAIN
# fc3 row-block shapes: rows x hidden x c (spd-census WH h=5 pair / 16 MSA; BH h=11 / 32).
SHAPES = {"wormhole": {"pair": (5 * 736, 1024, 256), "msa": (16 * 736, 512, 128)},
          "blackhole": {"pair": (11 * 736, 1024, 256), "msa": (32 * 736, 512, 128)}}[ARCH]


def cdiv(x, y):
    return -(-x // y)


def subblock(pm, pn):
    """Largest out subblock (h, w) with h*w <= 4 (fp32 dest half) dividing (pm, pn), widest first."""
    best = (1, 1)
    for w in range(1, 5):
        for h in range(1, 5):
            if h * w <= 4 and pm % h == 0 and pn % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def configs(M, K, N):
    mt, kt, nt = M // 32, K // 32, N // 32
    out = []
    for gx in sorted({GX, 4, 2, 1}):
        if gx > GX:
            continue
        pn = cdiv(nt, gx)
        if pn * gx != nt:
            continue
        for gy in sorted({GY}):
            pm = cdiv(mt, gy)
            for bw in (1, 2, 4, 8):
                if kt % bw:
                    continue
                sh, sw = subblock(pm, pn)
                out.append((f"2d:g{gx}x{gy}:bw{bw}:sb{sh}x{sw}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
                    per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)))
    for pm in sorted({cdiv(mt, GX * GY), cdiv(mt, GX * GY) + 1, 2, 3, 4}):
        if cdiv(mt, pm) > GX * GY:
            continue
        for bw in (2, 4, 8):
            if kt % bw:
                continue
            sh, sw = subblock(pm, nt)
            out.append((f"1d:pm{pm}:bw{bw}:sb{sh}x{sw}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
                out_block_h=pm, out_block_w=nt, per_core_M=pm, per_core_N=nt, fuse_batch=True,
                fused_activation=None, mcast_in0=False)))
    return out


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": OPENED,
       "grid": [GX, GY], "dtype": str(DT), "shapes": {}}
g = torch.Generator().manual_seed(0)
for name in a.shapes.split(","):
    M, K, N = SHAPES[name]
    xh = torch.randn(1, 1, M, K, generator=g) * torch.sigmoid(torch.randn(1, 1, M, K, generator=g))
    wh = torch.randn(K, N, generator=g) / K ** 0.5
    x = ttnn.from_torch(xh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    w = ttnn.from_torch(wh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.to_torch(x).double().view(M, K) @ ttnn.to_torch(w).double().view(K, N)
    rows = {}
    arms = [("auto", None)] + configs(M, K, N)
    for arm, pc in arms:
        kw = dict(compute_kernel_config=CKC, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kw.update(program_config=pc) if pc is not None else kw.update(core_grid=T.CORE_GRID_MAIN)
        fn = lambda: ttnn.linear(x, w, **kw)
        try:
            o = fn()
            got = ttnn.to_torch(o).double().view(M, N)
            ttnn.deallocate(o)
            row = dict(rel_rms=float((got - ref).norm() / ref.norm()),
                       digest=hashlib.sha256(got.float().numpy().tobytes()).hexdigest()[:16])
            row.update(timed(fn, a.batches))
        except Exception as e:  # noqa: BLE001  a config that does not fit L1 or is rejected is a result too
            row = dict(err=str(e).splitlines()[0][:200])
        rows[arm] = row
        print(json.dumps({name: {arm: row}}), flush=True)
    res["shapes"][name] = dict(shape=[M, K, N], arms=rows)
    ttnn.deallocate(x)
    ttnn.deallocate(w)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
