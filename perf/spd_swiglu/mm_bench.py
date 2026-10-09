"""The transition's three matmuls under every program config that fits: us per call, error vs float64, digest.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/mm_bench.py --out OUT.json [--b8] [--silu-f32] [--ops fc1,fc2,fc3]

The fold runs fc1 (+ fused silu), fc2 and fc3 through ttnn.linear(core_grid=CORE_GRID_MAIN), whose auto config takes
in0_block_w = 1 and out_block_h = per_core_M. At c730 on WH fc3 then runs 114 us a call against 61 us for fc1's
matmul of the same FLOPs (census). Arms:
  auto                     what the fold runs
  2d:gXxY:pmM:bwB:obhH     MatmulMultiCoreReuseMultiCastProgramConfig
  1d:pmM:bwB:obhH          MatmulMultiCoreReuseMultiCast1DProgramConfig, mcast_in0=False: every core owns per_core_M
                           rows and the whole weight is multicast once
out_block_h only reorders the drain, so an arm that differs from auto in it alone writes auto's bytes; in0_block_w
above 1 folds the K partials through packer_l1_acc in a different order and moves the last bit. Input in L1 (as in
the fold), fc1/fc2 out bf16 to L1, fc3 out bf16 to DRAM, Protenix's transition config (HiFi4, fp32 dest, packer L1
acc). --silu-f32 sets the silu_f32 lever (overlay at import, PACK-thread silu); --b8 the fast-mode bfp8 dtypes.
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
ap.add_argument("--ops", default="fc1,fc2,fc3")
ap.add_argument("--silu-f32", action="store_true")
a = ap.parse_args()
if a.silu_f32:
    os.environ["TT_BIO_LEVERS"] = "silu_f32"

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
with T.levers("silu_f32" if a.silu_f32 else ""):
    SILU_CKC = T.silu_ckc(CKC)  # what the fused silu site runs: approx mode from the lever
DT = ttnn.bfloat8_b if a.b8 else ttnn.bfloat16
GX, GY = T.COMPUTE_GRID_MAIN
# Row blocks of the fold (spd-census WH h=5 pair / 16 MSA; BH h=11 / 32): rows x c x hidden. pair384 is OpenDDE's
# c_z=384 pair transition (cs1 census, WH h=4 at W=736).
SHAPES = {"wormhole": {"pair": (5 * 736, 256, 1024), "msa": (16 * 736, 128, 512), "pair384": (4 * 736, 384, 1536)},
          "blackhole": {"pair": (11 * 736, 256, 1024), "msa": (32 * 736, 128, 512),
                        "pair384": (4 * 736, 384, 1536)}}[ARCH]
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)


def cdiv(x, y):
    return -(-x // y)


def subblock(h, w):
    """Largest out subblock (sh, sw) with sh*sw <= 4 (an fp32 dest half) dividing (h, w), widest first."""
    return max(((sh, sw) for sh in range(1, 5) for sw in range(1, 5)
                if sh * sw <= 4 and h % sh == 0 and w % sw == 0), key=lambda t: (t[0] * t[1], t[1]))


def configs(mt, kt, nt, act):
    out = []
    bws = [b for b in (1, 2, 3, 4, 6, 8) if kt % b == 0]
    for gx in sorted({GX, 6, 4, 3, 2}):  # 6 and 3: c=384's 12 output tiles do not split over 8
        if nt % gx:
            continue
        pn = nt // gx
        for pm in sorted({cdiv(mt, GY), cdiv(mt, GY - 1)}):
            gy = cdiv(mt, pm)
            if gy > GY:
                continue
            for obh in sorted({pm} | {h for h in (1, 2, 4, 5) if pm % h == 0}):
                for bw in bws:
                    sh, sw = subblock(obh, pn)
                    out.append((f"2d:g{gx}x{gy}:pm{pm}:bw{bw}:obh{obh}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh,
                        out_subblock_w=sw, out_block_h=obh, out_block_w=pn, per_core_M=pm, per_core_N=pn,
                        transpose_mcast=False, fused_activation=act, fuse_batch=True)))
    for pm in sorted({cdiv(mt, GX * GY), cdiv(mt, GX * GY) + 1, 4}):
        if cdiv(mt, pm) > GX * GY:
            continue
        for obh in sorted({pm, 1}):
            for bw in bws:
                sh, sw = subblock(obh, nt)
                out.append((f"1d:pm{pm}:bw{bw}:obh{obh}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY), in0_block_w=bw, out_subblock_h=sh,
                    out_subblock_w=sw, out_block_h=obh, out_block_w=nt, per_core_M=pm, per_core_N=nt,
                    fuse_batch=True, fused_activation=act, mcast_in0=False)))
    return out


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": OPENED,
       "grid": [GX, GY], "dtype": str(DT), "silu_f32": a.silu_f32, "shapes": {}}
g = torch.Generator().manual_seed(0)
for name in a.shapes.split(","):
    R, C, HID = SHAPES[name]
    for op in a.ops.split(","):
        M, K, N = (R, HID, C) if op == "fc3" else (R, C, HID)
        silu = op == "fc1"
        xh = torch.randn(1, 1, M, K, generator=g)
        if op == "fc3":
            xh = xh * torch.sigmoid(torch.randn(1, 1, M, K, generator=g))  # a silu * gate product
        wh = torch.randn(K, N, generator=g) / K ** 0.5
        x = ttnn.from_torch(xh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        w = ttnn.from_torch(wh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev)
        ref = ttnn.to_torch(x).double().view(M, K) @ ttnn.to_torch(w).double().view(K, N)
        if silu:
            ref = torch.nn.functional.silu(ref)
        ckc = SILU_CKC if silu else CKC
        mc = ttnn.DRAM_MEMORY_CONFIG if op == "fc3" else ttnn.L1_MEMORY_CONFIG
        odt = ttnn.bfloat16 if op == "fc3" else DT
        rows = {}
        for arm, pc in [("auto", None)] + configs(M // 32, K // 32, N // 32, SILU if silu else None):
            kw = dict(compute_kernel_config=ckc, dtype=odt, memory_config=mc)
            if pc is None:
                kw.update(core_grid=T.CORE_GRID_MAIN, activation="silu" if silu else None)
            else:
                kw.update(program_config=pc)
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
            print(json.dumps({f"{name}.{op}": {arm: row}}), flush=True)
        res["shapes"][f"{name}.{op}"] = dict(shape=[M, K, N], arms=rows)
        ttnn.deallocate(x)
        ttnn.deallocate(w)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
