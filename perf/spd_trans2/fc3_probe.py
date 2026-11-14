"""The transition's output projection (census "fc2", code fc3): K block x fidelity x fp32 dest, against float64.

    TT_VISIBLE_DEVICES=N python perf/spd_trans2/fc3_probe.py --out OUT.json [--shapes pair,msa] [--draws 8]

Per arm: us per call (warm, AICLK sampled), rel rms vs float64, and the count of outputs off float64 by more than
0.25 over `--draws` random draws. The last is the Wormhole fp32-dest erratum (state/spd-wherr.md: +-2 / +-4 on a
few outputs per 1e7 when more than one K tile accumulates in dest); bf16 rounding of these outputs stays < 0.05.
Each arm is also run on a narrower grid: `grid_same` says whether the bytes depend on the grid.
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
ap.add_argument("--draws", type=int, default=8)
ap.add_argument("--shapes", default="pair,msa")
ap.add_argument("--b8", action="store_true", help="fast mode's bfp8 hidden and weights")
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
    v = sorted(x for ts, r in samples if t0 <= ts <= max(t1, samples[-1][0]) for x in r.values())
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


def timed(fn, batches):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    t0 = time.monotonic()
    for _ in range(batches):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(10):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / 10 * 1e6)
    return dict(us_min=round(min(ts), 2), us_med=round(st.median(ts), 2), clock=clock(t0, time.monotonic()))


CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi3": ttnn.MathFidelity.HiFi3,
       "hifi4": ttnn.MathFidelity.HiFi4}


def ckc(fid, acc=True):
    return CKC_CLS(math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=True)


DT = ttnn.bfloat8_b if a.b8 else ttnn.bfloat16
GX, GY = T.COMPUTE_GRID_MAIN
# The fold's row blocks (spd-census r18 / bh17): rows x hidden x c.
SHAPES = {"wormhole": {"pair": (5 * 736, 1024, 256), "msa": (16 * 736, 512, 128)},
          "blackhole": {"pair": (11 * 736, 1024, 256), "msa": (32 * 736, 512, 128)}}[ARCH]


def cdiv(x, y):
    return -(-x // y)


def subblock(h, w):
    return max(((sh, sw) for sh in range(1, 5) for sw in range(1, 5)
                if sh * sw <= 4 and h % sh == 0 and w % sw == 0), key=lambda t: (t[0] * t[1], t[1]))


def pc_1d(mt, nt, bw, gx, gy, pm=None):
    pm = pm or cdiv(mt, gx * gy)
    sh, sw = subblock(pm, nt)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=nt, per_core_M=pm, per_core_N=nt, fuse_batch=True, fused_activation=None,
        mcast_in0=False)


def pc_2d(mt, nt, bw, gx, gy):
    pm, pn = cdiv(mt, gy), nt // gx
    sh, sw = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, cdiv(mt, pm)), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


def arms(mt, kt, nt):
    """(name, fidelity, fp32 acc, config factory on (gx, gy) or None for auto)."""
    out = []
    for fid in ("hifi2", "hifi3", "hifi4"):
        out.append((f"auto:{fid}", fid, True, None))
        for bw in (1, 2, 4, 8):
            if kt % bw == 0:
                out.append((f"1d:bw{bw}:{fid}", fid, True, lambda gx, gy, bw=bw: pc_1d(mt, nt, bw, gx, gy)))
        if nt % GX == 0:
            out.append((f"2d:bw8:{fid}", fid, True, lambda gx, gy: pc_2d(mt, nt, 8, gx, gy)))
    for fid in ("hifi2", "hifi3"):
        out.append((f"auto:{fid}:acc0", fid, False, None))
        out.append((f"1d:bw8:{fid}:acc0", fid, False, lambda gx, gy: pc_1d(mt, nt, 8, gx, gy)))
    return out


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": OPENED,
       "grid": [GX, GY], "dtype": str(DT), "shapes": {}}
for name in a.shapes.split(","):
    M, K, N = SHAPES[name]
    draws = []
    for d in range(a.draws):
        g = torch.Generator().manual_seed(1000 + d)
        xh = torch.randn(1, 1, M, K, generator=g) * torch.sigmoid(torch.randn(1, 1, M, K, generator=g))
        wh = torch.randn(K, N, generator=g) / K ** 0.5
        x = ttnn.from_torch(xh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev)  # DRAM; L1 copy per run
        w = ttnn.from_torch(wh, dtype=DT, layout=ttnn.TILE_LAYOUT, device=dev)
        ref = ttnn.to_torch(x).double().view(M, K) @ ttnn.to_torch(w).double().view(K, N)
        draws.append((x, w, ref))
    rows = {}
    for arm, fid, acc, mk in arms(M // 32, K // 32, N // 32):
        cfg = ckc(fid, acc)

        def run(x, w, gx=GX, gy=GY):
            xl = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # the fold's fc3 reads its input from L1
            try:
                return mm(xl, w, gx, gy)
            finally:
                ttnn.deallocate(xl)

        def mm(x, w, gx=GX, gy=GY):
            kw = dict(compute_kernel_config=cfg, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if mk is None:
                kw["core_grid"] = ttnn.CoreGrid(y=gy, x=gx)
            else:
                kw["program_config"] = mk(gx, gy)
            return ttnn.linear(x, w, **kw)

        try:
            bad = 0
            num = den = 0.0
            maxerr = 0.0
            digests = []
            for x, w, ref in draws:
                o = run(x, w)
                got = ttnn.to_torch(o).double().view(M, N)
                ttnn.deallocate(o)
                e = (got - ref).abs()
                bad += int((e > 0.25).sum())
                maxerr = max(maxerr, float(e.max()))
                num += float(((got - ref) ** 2).sum())
                den += float((ref ** 2).sum())
                digests.append(hashlib.sha256(got.float().numpy().tobytes()).hexdigest()[:16])
            row = dict(rel_rms=(num / den) ** 0.5, bad=bad, outputs=M * N * len(draws), max_err=maxerr,
                       digest=digests[0])
            try:  # the same arm on one grid row fewer: do the bytes depend on the grid?
                o = run(draws[0][0], draws[0][1], GX, GY - 1)
                row["grid_same"] = hashlib.sha256(ttnn.to_torch(o).float().view(M, N).numpy().tobytes()
                                                  ).hexdigest()[:16] == digests[0]
                ttnn.deallocate(o)
            except Exception as e:  # noqa: BLE001
                row["grid_same"] = f"err {str(e).splitlines()[0][:80]}"
            xl = ttnn.to_memory_config(draws[0][0], ttnn.L1_MEMORY_CONFIG)
            row.update(timed(lambda: mm(xl, draws[0][1]), a.batches))
            ttnn.deallocate(xl)
        except Exception as e:  # noqa: BLE001  a config that does not fit is a result too
            row = dict(err=str(e).splitlines()[0][:200])
        rows[arm] = row
        print(json.dumps({name: {arm: row}}), flush=True)
    res["shapes"][name] = dict(shape=[M, K, N], arms=rows)
    for x, w, _ in draws:
        ttnn.deallocate(x)
        ttnn.deallocate(w)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
